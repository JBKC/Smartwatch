'''
Training Step 2: Perform motion artifact removal on raw training data using linear adaptive filter
Saves down to new table within PostgreSQL database
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


def ma_removal(data_dict, sessions):
    '''
    Remove session-specific motion artifacts from raw PPG data by training on accelerometer_cnn
    Save down to dictionary "ppg_filt_dict"
    :param data_dict: dictionary containing ppg, acc, label and activity data for each session
    :param s: list of sessions
    :return: ppg_filt_dict: dictionary containing bvp (cleaned ppg) along with acc, label and activity
    '''

    # ppg_dalia_dict filtered for motion artifacts

    ppg_filt_dict = {f'{session}': {} for session in sessions}

    # initialise CNN model
    n_epochs = 1000
    model = AdaptiveLinearModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=1e-2)

    for s in sessions:

        X_BVP = []  # filtered PPG data

        # concatenate ppg + accelerometer signal data -> (n_windows, 4, 256)
        X = np.concatenate((data_dict[s]['ppg'], data_dict[s]['acc']), axis=1)
        act = data_dict[s]['activity']  # Activity labels

        # find indices of activity changes (marks batches)
        idx = np.argwhere(np.abs(np.diff(data_dict[s]['activity'])) > 0).flatten() +1

        # add indices of start and end points
        idx = np.insert(idx, 0, 0)
        idx = np.insert(idx, idx.size, data_dict[s]['label'].shape[0])

        initial_state = model.state_dict()

        # iterate over activities (batches)
        for i in range(idx.size - 1):

            model.load_state_dict(initial_state)

            # create batches
            X_batch = X[idx[i]: idx[i + 1], :, :]  # splice X into current activity

            # batch Z-normalisation
            X_batch, ms, stds = z_normalise(X_batch)

            X_batch = np.expand_dims(X_batch, axis=1)          # add channel dimension
            X_batch = torch.from_numpy(X_batch).float()

            # accelerometer data are inputs:
            x_acc = X_batch[:, :, 1:, :]                 # (batch_size, 1, 3, 256)
            # PPG data are targets:
            x_ppg = X_batch[:, :, :1, :]                 # (batch_size, 1, 1, 256)
            # activity-aware element

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

                # print(f'Session {s}, Batch: [{i + 1}/{idx.size - 1}], '
                #       f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

            # subtract the motion artifact estimate from raw signal to extract cleaned BVP
            with torch.no_grad():
                x_bvp = x_ppg[:, 0, 0, :] - model(x_acc)

            # get signal into original shape: (n_windows, 1, 256)
            x_bvp = torch.unsqueeze(x_bvp, dim=1).numpy()
            x_bvp = undo_normalisation(x_bvp, ms, stds)
            x_bvp = np.expand_dims(x_bvp[:,0,:], axis=1)            # keep only BVP (remove ACC)
            X_BVP.append(x_bvp)

        # add to dictionary
        X_BVP = np.concatenate(X_BVP, axis=0)                   # shape (n_windows, 1, 256)
        print(X_BVP.shape)

        # finally z-normalise each window individually to match inference
        X_BVP = window_z_normalise(X_BVP)

        ppg_filt_dict[s]['bvp'] = X_BVP
        ppg_filt_dict[s]['acc'] = data_dict[s]['acc']
        ppg_filt_dict[s]['label'] = data_dict[s]['label']
        ppg_filt_dict[s]['activity'] = data_dict[s]['activity']

        print(f"{s} shape: {ppg_filt_dict[s]['bvp'].shape}")

    # save dictionary
    with open('ppg_filt_dict', 'wb') as file:
        pickle.dump(ppg_filt_dict, file)
    print(f'Data dictionary saved to ppg_filt_dict')

    return

def main():

    # filter data and add to another SQL table
    database = "smartwatch_raw_data_all"
    extract_table = "session_data"
    save_table = "ma_filtered_data"

    # connect to database
    conn = psycopg2.connect(
        dbname=database,
        user="postgres",
        password="newpassword",
        host="localhost",
        port=5432
    )
    cur = conn.cursor()

    # pass individual windows through adaptive filter to clean PPG signal
    column_names = [desc[0] for desc in cur.description]

    # Print column names
    print("Column Names:", column_names)


if __name__ == '__main__':
    main()