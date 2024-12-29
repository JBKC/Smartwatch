'''
Training Step 1: Extract raw data, perform intial pre-processing, slice into windows, and upload to PostgreSQL database
'''

import pickle
import numpy as np
import pandas as pd
import os
import psycopg2
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

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

    # sampling rates
    fs = {
        'ppg': 32,                  # fs_ppg = 64 in paper but downsampled to match accelerometer
        'acc': 32,
        'activity': 4
    }

    n_windows = int(len(window_dict['label']))

    for k, f in fs.items():
        # can alternatively use skimage.util.shape.view_as_windows method

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

    return window_dict


def save_ppg_dalia(dir, conn, cur):
    '''
    ## TRAINING DATASET 1 - PPG-Dalia
    '''

    dataset = 'ppg_dalia'

    # iterate through sessions
    for s in os.listdir(f'{dir}/ppg+dalia'):
        if s == '.DS_Store':  # Skip .DS_Store
            continue
        with open(f'{dir}/ppg+dalia/{s}/{s}.pkl', 'rb') as file:

            print(f'extracting: {dataset}, {s}')

            data = pickle.load(file, encoding='latin1')

            # get raw data from pkl file
            ppg = data['signal']['wrist']['BVP'][::2]  # downsample PPG to match fs_acc
            acc = data['signal']['wrist']['ACC']
            activity = data['activity']
            label = data['label']  # ground truth EEG

            # alignment corrections & filter data for consistency
            ppg = butter_filter(signal=ppg[38:].T, btype='bandpass', lowcut=0.5, highcut=15)
            acc = butter_filter(signal=acc[:-38, :].T, btype='lowpass', highcut=15)
            activity = activity[:-1].T
            label = label[:-1]

            # print(ppg.shape)
            # print(acc.shape)
            # print(activity.shape)
            # print(label.shape)

            # add to dictionary and window
            window_dict = {
                'ppg': ppg,
                'acc': acc,
                'activity': activity,
                'label': label
            }
            window_dict = window_data(window_dict)

            print(window_dict['ppg'].shape)
            print(window_dict['acc'].shape)
            print(window_dict['activity'].shape)
            print(window_dict['label'].shape)

            rows = [
                (
                    dataset,
                    s,
                    window_dict['ppg'][i,:].tolist(),
                    window_dict['acc'][i,:,:].tolist(),
                    int(window_dict['activity'][i]),
                    window_dict['label'][i]
                )
                for i in range(len(label))
            ]

            # Insert into the SQL database
            cur.executemany("""
                INSERT INTO session_data (dataset, session_number, ppg, acc, activity, label)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, rows)

            conn.commit()
            print(f'saved: {dataset}, {s}')


def save_wrist_ppg(dir, conn, cur):
    '''
    ## TRAINING DATASET 2 - Wrist PPG
    '''

    dataset = 'wrist_ppg'

    # iterate through sessions
    for s in os.listdir(f'{dir}/ppg+dalia'):
        if s == '.DS_Store':  # Skip .DS_Store
            continue
        with open(f'{dir}/ppg+dalia/{s}/{s}.pkl', 'rb') as file:

            print(f'extracting {s}')

            data = pickle.load(file, encoding='latin1')

    # from .hea file: 0 = ecg, 1 = ppg, 2-4 = gyro, 5-7 = 2g accelerometer, 8-10 = 16g accelerometer
    df = pd.DataFrame({
        'ppg': record.adc()[:, 0],
        'x': record.adc()[:, 5],
        'y': record.adc()[:, 6],
        'z': record.adc()[:, 7]
    })
    ppg_data = df['ppg']
    x_data = df['x']
    y_data = df['y']
    z_data = df['z']

    with open('/Users/jamborghini/Documents/PYTHON/Fatigue Model/' + session_name +'_heart_rate_wrist_ppg.pkl', 'rb') as file:
        ecg_ground_truth = pickle.load(file, encoding='latin1')

    fs_ppg = 256  # from the paper
    fs_acc = 256
    num_ppg = len(ppg_data)
    num_acc = len(x_data)
    time_analyse_seconds = len(ppg_data) / fs_ppg  # this is the total time in seconds

    return ppg_data, x_data, y_data, z_data, ecg_ground_truth, fs_ppg, fs_acc, num_ppg, num_acc

## TRAINING DATASET 3 - preprocessing WESAD
def preprocess_data_wesad(session_name):

    ppg_file = '/Users/jamborghini/Documents/PYTHON/TRAINING DATA - WESAD/'+session_name+'/'+session_name+'_E4_Data/BVP.csv'
    acc_file = '/Users/jamborghini/Documents/PYTHON/TRAINING DATA - WESAD/'+session_name+'/'+session_name+'_E4_Data/ACC.csv'

    df_ppg = pd.read_csv(ppg_file)
    ppg_data = df_ppg.iloc[1:, 0]
    df_acc = pd.read_csv(acc_file)
    x_data = df_acc.iloc[1:, 0]
    y_data = df_acc.iloc[1:, 1]
    z_data = df_acc.iloc[1:, 2]
    print(len(ppg_data))
    print(len(x_data))

    with open('/Users/jamborghini/Documents/PYTHON/Fatigue Model/' + session_name +'_heart_rate_wesad.pkl', 'rb') as file:
        ecg_ground_truth = pickle.load(file, encoding='latin1')

    fs_ppg = 64  # Hz
    fs_acc = 32  # Hz
    num_ppg = len(ppg_data)
    num_acc = len(x_data)
    time_analyse_seconds = len(ppg_data) / fs_ppg  # this is the total time in seconds
    print(time_analyse_seconds)

    return ppg_data, x_data, y_data, z_data, ecg_ground_truth, fs_ppg, fs_acc, num_ppg, num_acc


def main():

    # connect to PostgreSQL database
    conn = psycopg2.connect(
        dbname="smartwatch_raw_data_all",
        user="postgres",
        password="newpassword",
        host="localhost",
        port=5432
    )
    cur = conn.cursor()

    # extract 3 datasets and add to PostgreSQL database
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = root_dir+'/raw_data'

    save_ppg_dalia(data_dir, conn, cur)
    # save_wrist_ppg(data_dir, conn, cur)


if __name__ == '__main__':
    main()
