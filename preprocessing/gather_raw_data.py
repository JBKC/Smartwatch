'''
Training Step 1: Extract raw data, perform initial pre-processing, slice into windows, and upload to PostgreSQL database
'''

import pickle
import numpy as np
import os
import psycopg2
from scipy.signal import butter, filtfilt, resample
import wfdb
import matplotlib.pyplot as plt
import peak_detection_qrs

activity_mapping = {
    0: 'none',
    1: "sitting still",
    2: "stairs",
    3: "table football",
    4: "cycling",
    5: "driving",
    6: "lunch break",
    7: "walking",
    8: "working at desk",
    9: "running"
}

def butter_filter(signal, btype, lowcut=None, highcut=None, fs=32, order=5):
    """
    Applies Butterworth filter
    :param signal: input signal of shape (n_channels, n_samples)
    :return smoothed: smoothed signal of shape (n_channels, n_samples)
    """

    nyquist = 0.5 * fs

    if btype == 'bandpass':
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype=btype)
    elif btype == 'lowpass':
        high = highcut / nyquist
        b, a = butter(order, high, btype=btype)
    elif btype == 'highpass':
        low = lowcut / nyquist
        b, a = butter(order, low, btype=btype)

    # apply filter using filtfilt (zero-phase filtering)
    filtered = np.array([filtfilt(b, a, channel) for channel in signal])

    return filtered

def window_data(window_dict):
    '''
    Segment data into windows of 8 seconds with 2 second overlap. Only used when saving down raw data for first time
    :param window_dict: dictionary with all signals in arrays for given session
    :return: dictionary of windowed signals containing X and Y data
        ppg.shape = (n_windows, 256)
        acc.shape = (n_windows, 3, 256)
        labels.shape = (n_windows,)
        activity.shape = (n_windows,)
    '''

    # sampling rates (PPG dalia reference)
    fs = {
        'ppg': 32,
        'acc': 32,
        'activity': 4
    }

    # align n_windows
    window_size = 8*32
    step_size = 2*32
    data_size = window_dict['ppg'].shape[1]
    n_windows = (data_size - window_size) // step_size + 1
    n_windows = min(len(window_dict['label']), n_windows)

    # Truncate label array to match n_windows
    if 'label' in window_dict:
        window_dict['label'] = window_dict['label'][:n_windows]

    for k, f in fs.items():
        if k not in window_dict:
            continue

        window = 8*f                        # size of window
        step = 2*f                          # size of step
        data = window_dict[k]

        if k == 'ppg':
            # (1, n_samples) -> (n_windows, 256)
            window_dict[k] = np.zeros((n_windows, window))
            for i in range(n_windows):
                start = i * step
                end = start + window
                window_dict[k][i, :] = data[:, start:end]

        if k == 'acc':
            # (3, n_samples) -> (n_windows, 3, 256)
            window_dict[k] = np.zeros((n_windows, 3, window))
            for i in range(n_windows):
                start = i * step
                end = start + window
                window_dict[k][i, :, :] = data[:, start:end]

        if k == 'activity':
            window_dict[k] = np.zeros((n_windows,))
            for i in range(n_windows):
                start = i * step
                end = start + window
                window_dict[k][i] = data[0, start:end][0]             # take first value as value of whole window

    return window_dict, n_windows

def save_ppg_dalia(dir, table, conn, cur):
    '''
    ## TRAINING DATASET 1 - PPG-Dalia
    '''

    dataset = 'ppg_dalia'

    # iterate through sessions
    for s in os.listdir(f'{dir}/ppg+dalia'):
        if s == '.DS_Store':  # Skip .DS_Store
            continue
        with open(f'{dir}/ppg+dalia/{s}/{s}.pkl', 'rb') as file:

            print(f'extracting >> {dataset} | {s}')

            data = pickle.load(file, encoding='latin1')

            # get raw data from pkl file
            ppg = data['signal']['wrist']['BVP'][::2]  # downsample PPG to match fs_acc
            acc = data['signal']['wrist']['ACC']
            activity = data['activity']
            label = data['label']  # ground truth EEG

            # print(ppg.shape)                            # (n_samples, 1)
            # print(acc.shape)                            # (n_samples, 3)
            # print(activity.shape)                       # (n_samples, 1)
            # print(label.shape)                          # (n_samples,)

            # alignment corrections & filter data for consistency
            ppg = butter_filter(signal=ppg[38:].T, btype='bandpass', lowcut=0.5, highcut=15)
            acc = butter_filter(signal=acc[:-38, :].T, btype='lowpass', highcut=15)
            activity = activity[:-1].T
            label = label[:-1]

            # plt.plot(ppg[0, :])
            # plt.show()
            # plt.plot(acc[0, :])
            # plt.show()

            # print(ppg.shape)                            # (1, n_samples)
            # print(acc.shape)                            # (3, n_samples)
            # print(activity.shape)                       # (1, n_samples)
            # print(label.shape)                          # (n_samples,)

            # add to dictionary and window
            window_dict = {
                'ppg': ppg,
                'acc': acc,
                'activity': activity,
                'label': label
            }
            window_dict, _ = window_data(window_dict)

            # remove transient periods (no activity label)
            indices = window_dict['activity'] != 0  # boolean mask
            window_dict['ppg'] = window_dict['ppg'][indices, :]
            window_dict['acc'] = window_dict['acc'][indices, :, :]
            window_dict['activity'] = window_dict['activity'][indices]
            window_dict['label'] = window_dict['label'][indices]

            print(window_dict['ppg'].shape)             # (n_samples, 256)
            print(window_dict['acc'].shape)             # (n_samples, 3, 256)
            print(window_dict['activity'].shape)        # (n_samples,)
            print(window_dict['label'].shape)           # (n_samples,)

            # sense check
            for key, value in activity_mapping.items():
                count = np.sum(window_dict['activity'] == key)
                print(value, count)

            # Insert into the SQL database
            rows = [
                (
                    dataset,
                    s,
                    window_dict['ppg'][i,:].tolist(),
                    window_dict['acc'][i,:,:].tolist(),
                    int(window_dict['activity'][i]),
                    window_dict['label'][i]
                )
                for i in range(window_dict['label'].shape[0])           # n_windows with transient periods removed
            ]

            query = f"""
                INSERT INTO {table} (dataset, session_number, ppg, acc, activity, label)
                VALUES (%s, %s, %s, %s, %s, %s)
            """

            cur.executemany(query, rows)
            conn.commit()


            print(f'saved >> {dataset} | {s}')

    print(f'extraction completed: {dataset},')

    return


def save_wrist_ppg(dir, table, conn, cur):
    '''
    ## TRAINING DATASET 2 - Wrist PPG
    '''

    dataset = 'wrist_ppg'

    # iterate through session files
    sessions = [f for f in os.listdir(f'{dir}/wrist+ppg') if f.endswith('.hea')]

    for s in sessions:
        # remove extension
        s, _ = os.path.splitext(s)

        record = wfdb.rdrecord(f'{dir}/wrist+ppg/{s}')
        print(f'extracting >> {dataset} | {s}')

        # .hea file format: 0 = ecg, 1 = ppg, 5-7 = 2g accelerometer
        # all recorded at 256Hz (downsample to 32Hz)
        ecg = record.adc()[:,0]
        ppg = np.expand_dims(record.adc()[:, 1][::8], axis=-1)
        acc = np.stack((record.adc()[:, 5][::8], record.adc()[:, 6][::8], record.adc()[:, 7][::8]), axis=-1)

        # print(ppg.shape)  # (n_samples, 1)
        # print(acc.shape)  # (n_samples, 3)

        ppg = butter_filter(signal=ppg.T, btype='bandpass', lowcut=0.5, highcut=15)
        acc = butter_filter(signal=acc.T, btype='lowpass', highcut=15)

        # plt.plot(ppg[0, :])
        # plt.show()
        # plt.plot(acc[0, :])
        # plt.show()


        print(ppg.shape)  # (1, n_samples)
        print(acc.shape)  # (3, n_samples)

        # generate labels using peak detection algorithm on ECG
        label = peak_detection_qrs.main(ecg=ecg,fs=256)

        # add to dictionary and window
        window_dict = {
            'ppg': ppg,
            'acc': acc,
            'label': label
        }
        window_dict, n_windows = window_data(window_dict)

        print(window_dict['ppg'].shape)  # (n_samples, 256)
        print(window_dict['acc'].shape)  # (n_samples, 3, 256)
        print(window_dict['label'].shape)  # (n_samples,)

        # activity mapping
        if "bike" in s.lower():
            activity = 4
        elif "walk" in s.lower():
            activity = 7
        elif "run" in s.lower():
            activity = 9

        print(int(activity))

        # Insert into the SQL database
        rows = [
            (
                dataset,
                s,
                window_dict['ppg'][i, :].tolist(),
                window_dict['acc'][i, :, :].tolist(),
                int(activity),
                window_dict['label'][i]
            )
            for i in range(n_windows)
        ]

        query = f"""
            INSERT INTO {table} (dataset, session_number, ppg, acc, activity, label)
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        cur.executemany(query, rows)
        conn.commit()
        print(f'saved >> {dataset} | {s}')

    print(f'extraction completed: {dataset},')

    return


def save_wesad(dir, table, conn, cur):
    '''
    ## TRAINING DATASET 3 - PPG-Dalia
    '''

    dataset = 'wesad'

    # iterate through sessions
    for s in os.listdir(f'{dir}/wesad'):
        if s == '.DS_Store':  # Skip .DS_Store
            continue
        with open(f'{dir}/wesad/{s}/{s}.pkl', 'rb') as file:

            print(f'extracting >> {dataset} | {s}')

            data = pickle.load(file, encoding='latin1')

            # get raw data from pkl file
            ppg = data['signal']['wrist']['BVP'][::2]  # downsample PPG to match fs_acc
            acc = data['signal']['wrist']['ACC']
            ecg = np.squeeze(data['signal']['chest']['ECG'][::7])  # downsample for efficiency

            ppg = butter_filter(signal=ppg[38:].T, btype='bandpass', lowcut=0.5, highcut=15)
            acc = butter_filter(signal=acc[:-38, :].T, btype='lowpass', highcut=15)

            # plt.plot(ppg[0, :])
            # plt.show()
            # plt.plot(acc[0, :])
            # plt.show()

            # generate labels using peak detection algorithm on ECG
            label = peak_detection_qrs.main(ecg=ecg,fs=100)

            # print(ppg.shape)                            # (1, n_samples)
            # print(acc.shape)                            # (3, n_samples)
            # print(label.shape)                          # (n_samples,)

            # add to dictionary and window
            window_dict = {
                'ppg': ppg,
                'acc': acc,
                'label': label
            }
            window_dict, n_windows = window_data(window_dict)

            print(window_dict['ppg'].shape)  # (n_samples, 256)
            print(window_dict['acc'].shape)  # (n_samples, 3, 256)
            print(window_dict['label'].shape)  # (n_samples,)

            # activity mapping (set all to working at desk)
            activity = int(8)

            # Insert into the SQL database
            rows = [
                (
                    dataset,
                    s,
                    window_dict['ppg'][i, :].tolist(),
                    window_dict['acc'][i, :, :].tolist(),
                    activity,                             # set activity to working at desk
                    window_dict['label'][i]
                )
                for i in range(n_windows)
            ]

            query = f"""
                INSERT INTO {table} (dataset, session_number, ppg, acc, activity, label)
                VALUES (%s, %s, %s, %s, %s, %s)
            """

            cur.executemany(query, rows)
            conn.commit()
            print(f'saved >> {dataset} | {s}')

    print(f'extraction completed: {dataset},')

    return


def main():

    # pre-created database & table within PostgreSQL
    database = "smartwatch_raw_data_all"
    table = "session_data"

    # connect to database
    conn = psycopg2.connect(
        dbname=database,
        user="postgres",
        password="newpassword",
        host="localhost",
        port=5432
    )
    cur = conn.cursor()

    # extract 3 datasets and add to PostgreSQL database
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = root_dir+'/raw_data'

    ###
    save_ppg_dalia(data_dir, table, conn, cur)
    save_wrist_ppg(data_dir, table, conn, cur)
    save_wesad(data_dir, table, conn, cur)

    print(f'Data extraction complete.')

    cur.close()
    conn.close()

if __name__ == '__main__':
    main()
