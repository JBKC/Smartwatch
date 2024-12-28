'''
Extract raw data and upload to PostgreSQL database
'''

import pickle
import numpy as np
import pandas as pd
import os
import psycopg2

def save_ppg_dalia(dir, cur):
    '''
    ## TRAINING DATASET 1 - PPG-Dalia
    '''

    # iterate through sessions
    for s in os.listdir(f'{dir}/ppg+dalia'):
        if s == '.DS_Store':  # Skip .DS_Store
            continue
        with open(f'{dir}/ppg+dalia/{s}/{s}.pkl', 'rb') as file:

            print(f'saving {s}')

            data = pickle.load(file, encoding='latin1')

            # get raw data from pkl file
            ppg = np.squeeze(data['signal']['wrist']['BVP'][::2])  # downsample PPG to match fs_acc
            acc = data['signal']['wrist']['ACC']
            activity = np.squeeze(data['activity'])
            label = np.squeeze(data['label'])  # ground truth EEG

            print(ppg.shape)
            print(acc.shape)
            print(activity.shape)
            print(label.shape)


            # alignment corrections
            data_dict[s]['ppg'] = data_dict[s]['ppg'][38:, :].T  # (1, n_samples)
            data_dict[s]['acc'] = data_dict[s]['acc'][:-38, :].T  # (3, n_samples)
            data_dict[s]['label'] = data_dict[s]['label'][:-1]  # (n_windows,)
            data_dict[s]['activity'] = data_dict[s]['activity'][:-1, :].T  # (1, n_samples)

            print(data_dict[s]['ppg'].shape)
            print(data_dict[s]['activity'])

            # Insert into the SQL database
            cur.execute("""
                INSERT INTO raw_ppg_dalia (session_name, ppg, acc_x, acc_y, acc_z, label, activity)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (s, ppg, acc_x, acc_y, acc_z, label, activity))

            conn.commit()
            print(f'Inserted session: {s}')



## TRAINING DATASET 2 - preprocessing Wrist PPG
def preprocess_data_wrist_ppg(session_name):
    file_path = '/Users/jamborghini/Documents/PYTHON/TESTING DATA - Wrist PPG Dataset/'
    file_name = file_path + session_name
    record = wfdb.rdrecord(file_name)

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

    save_ppg_dalia(data_dir, cur)


if __name__ == '__main__':
    main()
