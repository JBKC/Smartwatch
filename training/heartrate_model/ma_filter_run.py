'''
Training Step 2: Perform motion artifact removal on raw training data using linear adaptive filter
Saves filtered data to new table within PostgreSQL database
Saves individual model parameter sets for each activity
'''

import pickle
import numpy as np
from ma_filter import AdaptiveLinearModel
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import butter, filtfilt
import psycopg2

activity_mapping = {
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

def z_normalise(X):
    '''
    Z-normalises data for all windows, across each channel, using vectorisation
    :param X: of shape (n_windows, 4, 256)
    :return:
        X_norm: of shape (n_windows, 4, 256)
        ms (means): of shape (n_windows, 4)
        stds (standard deviations) of shape (n_windows, 4)
    '''

    # calculate mean and stdev for each channel in each window - creates shape (n_windows, 4, 1)
    ms = X.mean(axis=2, keepdims=True)
    stds = X.std(axis=2, keepdims=True)

    # Z-normalisation
    X_norm = (X - ms) / np.where(stds != 0, stds, 1)

    return X_norm, ms.squeeze(axis=2), stds.squeeze(axis=2)

def undo_normalisation(X_norm, ms, stds):
    '''
    Transform cleaned PPG signal back into original space following filtering
    :params:
        X_norm: of shape (n_windows, 1, 256)
        ms (means): of shape (n_windows, 4)
        stds (standard deviations) of shape (n_windows, 4)
    :return:
        X: of shape (n_windows, 1, 256)
    '''

    ms_reshaped = ms[:, :, np.newaxis]
    stds_reshaped = stds[:, :, np.newaxis]

    return (X_norm * np.where(stds_reshaped != 0, stds_reshaped, 1)) + ms_reshaped

def train_ma_filter(cur, conn, acts, batch_size):
    '''
    Uses ma_filter architecture to remove motion artifacts from raw PPG signal by activity
    '''

    def fetch_activity_data(act, batch_size):
        '''
        Fetch all data for given activity in batches
        '''

        query = f"SELECT * FROM session_data WHERE activity = %s LIMIT %s OFFSET %s;"

        offset = 0
        while True:
            # extract data in batches of size limit
            cur.execute(query, (act, batch_size, offset))
            rows = cur.fetchall()
            if not rows:
                break

            # save row data as dictionary
            columns = [desc[0] for desc in cur.description]
            batch_dict = [dict(zip(columns, row)) for row in rows]
            yield batch_dict
            # produces list (len=batch_size) of 2D tuples
            offset += batch_size

    # train by activity
    for act in acts:

        # initialise activity-specific filter model
        n_epochs = 1000
        model = AdaptiveLinearModel()
        optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=1e-2)

        print(f"Training filter for {activity_mapping[act]}...")
        for batch in fetch_activity_data(act, batch_size):
            ppg = np.expand_dims(np.array([row['ppg'] for row in batch]), axis=1)
            acc = np.array([row['acc'] for row in batch])
            # print(ppg.shape)                # (batch_size, 1, 256)
            # print(acc.shape)                # (batch_size, 3, 256)

            X_BVP = []  # filtered PPG data

            # concatenate ppg + accelerometer signal data -> (n_windows, 4, 256)
            X = np.concatenate((ppg, acc), axis=1)
            # print(X.shape)                  # (batch_size, 4, 256)

            # batch Z-normalisation
            X, ms, stds = z_normalise(X)

            X = np.expand_dims(X, axis=1)          # add channel dimension
            X = torch.from_numpy(X).float()

            # accelerometer data == inputs:
            x_acc = X[:, :, 1:, :]                 # (batch_size, 1, 3, 256)
            print(x_acc.shape)
            # PPG data == targets:
            x_ppg = X[:, :, :1, :]                 # (batch_size, 1, 1, 256)
            print(x_ppg.shape)

            # training loop
            for epoch in range(n_epochs):

                # forward pass through CNN to get x_ma (motion artifact estimate)
                x_ma = model(x_acc)
                # compute loss against raw PPG data
                loss = model.adaptive_loss(y_true=x_ppg, y_pred=x_ma)
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f'Activity {activity_mapping[act]},'
                      f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

            # subtract the motion artifact estimate from raw signal to extract cleaned BVP
            with torch.no_grad():
                x_bvp = x_ppg[:, 0, 0, :] - model(x_acc)

            # get signal into original shape (n_windows, 1, 256) and de-normalise
            x_bvp = torch.unsqueeze(x_bvp, dim=1).numpy()
            x_bvp = undo_normalisation(x_bvp, ms, stds)
            x_bvp = np.expand_dims(x_bvp[:,0,:], axis=1)            # keep only BVP (remove ACC)
            X_BVP.append(x_bvp)

            # save to new SQL table
            query = """
                INSERT INTO ma_filtered_data (row['dataset'], row['session_number'], %s
                 row['acc'], row['activity'], row['label'])
                VALUES (%s, %s, %s, %s, %s, %s)
            """

            cur.executemany(query, (X_BVP))
            conn.commit()

            print(f"Filtered data saved for {activity_mapping[act]}...")

    return

def main():

    def get_activities(cur):
        cur.execute("SELECT DISTINCT activity FROM session_data")
        activities = cur.fetchall()
        return [activity[0] for activity in activities]

    # connect to raw database
    database = "smartwatch_raw_data_all"
    conn = psycopg2.connect(
        dbname=database,
        user="postgres",
        password="newpassword",
        host="localhost",
        port=5432
    )
    cur = conn.cursor()

    # get unique activities
    acts = get_activities(cur)

    # train filters & filter data
    batch_size=256
    train_ma_filter(cur, conn, acts, batch_size)

    cur.close()
    conn.close()

if __name__ == '__main__':
    main()