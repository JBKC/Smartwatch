'''
Training Step 3: Main supervised attention-based model for extracting heartrate from filtered BVP
'''

import numpy as np
from sklearn.utils import shuffle
import time
import psycopg2
import os
# from temporal_attention_model import TemporalAttentionModel
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import psutil
from wandb_logger import WandBLogger


def temporal_pairs(x, labels):
    '''
    Create temporal pairs between adjacent windows for a single session
    :param x: all ppg windows for a single session of shape (n_windows, 256)
    :param y: all labels for a single session of shape (n_windows,)
    '''

    # pair adjacent windows in format (i, i-1)
    x_pairs = (np.expand_dims(x[1:, :], axis=-1), np.expand_dims(x[:-1, :], axis=-1))
    x_pairs = np.concatenate(x_pairs, axis=-1)
    y = labels[1:]

    return x_pairs, y

def data_generator(cur, groups, test_idx):
    '''
    Extracts data from SQL table, divides into LOSO groups and creates temporal pairs
    :param groups: list of sessions divided into groups
    :param test_idx: index of the current test session
    '''

    # extract training sessions (!= test session)
    train_sessions = [(dataset, session) for i, group in enumerate(groups) if i!=test_idx
                      for dataset, session in group]
    test_sessions = groups[test_idx]

    print(len(train_sessions))
    print(len(test_sessions))

    # get temporal pairings for training data
    x_train, y_train = [],[]
    x_test, y_test = [],[]

    # extract temporal pairs for training data
    for dataset, session in train_sessions:

        query = f"SELECT ppg FROM ma_filtered_data WHERE dataset=%s AND session_number=%s;"
        cur.execute(query, (dataset,session))
        ppg_windows = np.array([row[0] for row in cur.fetchall()])

        query = f"SELECT label FROM ma_filtered_data WHERE dataset=%s AND session_number=%s;"
        cur.execute(query, (dataset,session))
        labels = np.array([row[0] for row in cur.fetchall()])

        print(dataset, session, ppg_windows.shape, labels.shape)
        x_pairs, y = temporal_pairs(ppg_windows, labels)
        # print(x_pairs.shape)  # concatenated pairs of shape (n_windows, n_samples, 2)
        # print(y.shape)

        x_train.append(x_pairs)
        y_train.append(y)

    # extract temporal pairs for test data
    for dataset, session in test_sessions:

        query = f"SELECT ppg FROM ma_filtered_data WHERE dataset=%s AND session_number=%s;"
        cur.execute(query, (dataset,session))
        ppg_windows = np.array([row[0] for row in cur.fetchall()])

        query = f"SELECT label FROM ma_filtered_data WHERE dataset=%s AND session_number=%s;"
        cur.execute(query, (dataset,session))
        labels = np.array([row[0] for row in cur.fetchall()])

        x_pairs, y = temporal_pairs(ppg_windows, labels)

        x_test.append(x_pairs)
        y_test.append(y)

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    yield x_train, y_train, x_test, y_test


def train_model(cur, conn, datasets, batch_size, n_epochs, lr):

    def torch_convert(X, y, batch_size=10):
        '''
        Necessary when converting large amounts of numpy data to torch data in one go
        '''

        X_batches = []
        y_batches = []

        # Process in batches
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # Convert each batch to PyTorch tensor
            X_batches.append(torch.tensor(X_batch, dtype=torch.float32))
            y_batches.append(torch.tensor(y_batch, dtype=torch.float32))

        # Concatenate all batches
        X_tensor = torch.cat(X_batches, dim=0)
        y_tensor = torch.cat(y_batches, dim=0)

        return X_tensor, y_tensor

    def NLL(dist, y):
        '''
        Negative log likelihood loss of observation y, given distribution dist
        :param dist: predicted Gaussian distribution
        :param y: ground truth label
        :return: NLL for each window
        '''

        return -dist.log_prob(y)

    def create_groups(datasets,n_groups):
        '''
        Create grouped LOSO splits for training
        :return: list of groups
        '''

        all_sessions = []

        for dataset in datasets:
            query = f"SELECT DISTINCT session_number FROM ma_filtered_data WHERE dataset=%s;"
            cur.execute(query, (dataset,))
            sessions = [row[0] for row in cur.fetchall()]
            all_sessions.extend([(dataset, session) for session in sessions])

        np.random.shuffle(all_sessions)
        groups = np.array_split(all_sessions, n_groups)

        return groups

    # predefine the groups for LOSO split
    groups = create_groups(datasets,n_groups=5)

    # dynamically fetch training and test data
    for test_idx in range(len(groups)):
        print(f"LOSO Fold {test_idx + 1} of {len(groups)}")

        # use generator to fetch training & test data for given fold
        for x_train, y_train, x_test, y_test in data_generator(cur, groups, test_idx):
            print(f"Train: {x_train.shape}, {y_train.shape}")
            print(f"Test: {x_test.shape}, {y_test.shape}")

            x_train, y_train = torch_convert(x_train, y_train)
            x_test, y_test = torch_convert(x_test, y_test)
            
            print(f"Train: {x_train.shape}, {y_train.shape}")
            print(f"Test: {x_test.shape}, {y_test.shape}")


    # initialise model
    patience = 10               # early stopping parameter
    n_splits = 4



    # LOSO splits
    ids = shuffle(list(range(len(sessions))))       # index each session
    splits = np.array_split(ids, n_splits)

    start_time = time.time()

    # outer LOSO split for training data
    for split_idx, split in enumerate(splits):

        # # skip already-processed splits
        # if split_idx <= last_split_idx:
        #     continue

        # set training data (current split = testing/validation data)
        train_idxs = np.array([i for i in ids if i not in split])

        X_train = np.concatenate([x[i] for i in train_idxs], axis=0)
        y_train = np.concatenate([y[i] for i in train_idxs], axis=0)

        # compress representation
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)

        # Convert the dataset
        X_train, y_train = torch_convert(X_train, y_train)

        # create TensorDataset and DataLoader for batching
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # inner LOSO split for testing & validation data
        for session_idx, s in enumerate(split):

            # # skip already-processed sessions in current split
            # if split_idx == last_split_idx and session_idx <= last_session_idx:
            #     continue

            # set test data as the current session s within the current split
            X_test = x[s]
            y_test = y[s]
            X_test = X_test.astype(np.float32)
            y_test = y_test.astype(np.float32)
            X_test, y_test = torch_convert(X_test, y_test)

            # set validation data (remainder of current split)
            val_idxs = np.array([j for j in split if j != s])
            X_val = np.concatenate([x[j] for j in val_idxs], axis=0)
            y_val = np.concatenate([y[j] for j in val_idxs], axis=0)
            X_val = X_val.astype(np.float32)
            y_val = y_val.astype(np.float32)
            X_val, y_val = torch_convert(X_val, y_val)

            print(f"X_train shape: {X_train.shape}")
            print(f"X_val shape: {X_val.shape}")
            print(f"X_test shape: {X_test.shape}")

            # train separate model for each test session
            model = TemporalAttentionModel()
            optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08)

            # load checkpoint if available
            checkpoint_path = f'../models/temporal_attention_model_session_S{s+1}.pth'

            try:
                checkpoint = torch.load(checkpoint_path)

                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                best_val_loss = checkpoint['best_val_loss']
                counter = checkpoint['counter']
                saved_splits = checkpoint['saved_splits'],
                processed_splits = checkpoint['processed_splits']
                last_split_idx = checkpoint['last_split_idx']
                last_session = checkpoint['last_session']
                last_session_idx = checkpoint['last_session_idx']
                print(f"Checkpoint found, resuming from Split {last_split_idx + 1}, Session {last_session + 1}")

            except FileNotFoundError:
                print("No checkpoint found, training from scratch")
                best_val_loss = float('inf')  # early stopping parameter
                counter = 0  # early stopping parameter
                processed_splits = []  # track each split as they are processed
                last_split_idx = -1
                last_session_idx = -1
                epoch = -1

            print(
                f"System status before training: CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%")

            # training loop
            for epoch in range(epoch +1, n_epochs):

                print(f'Training started: test session = S{s+1}')

                model.train()

                # create training batches of windows to pass through model
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

                    optimizer.zero_grad()

                    # prep data for model input - shape is (batch_size, n_channels, sequence_length) = (256, 1, 256)
                    x_cur = X_batch[:, :, 0].unsqueeze(1)
                    x_prev = X_batch[:, :, -1].unsqueeze(1)

                    # forward pass through model (convolutions + attention + probabilistic)
                    dist = model(x_cur, x_prev)

                    # calculate training loss on distribution
                    loss = NLL(dist, y_batch).mean()
                    loss.backward()
                    optimizer.step()

                    print(f'Test session: S{s + 1}, Batch: [{batch_idx + 1}/{len(train_loader)}], '
                          f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {loss.item():.4f}')

                # validation on whole validation set after each epoch
                model.eval()

                with torch.no_grad():
                    val_dist = model(X_val[:,:,0].unsqueeze(1), X_val[:,:,-1].unsqueeze(1))
                    val_loss = NLL(val_dist, y_val).mean()          # average validation across all windows

                    print(f'Test session: S{s + 1}, Epoch [{epoch + 1}/{n_epochs}], Validation Loss: {val_loss.item():.4f}')

                # early stopping criteria
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0

                    # save down checkpoint of current best model state
                    checkpoint = {
                        'model_state_dict': model.state_dict(),  # model weights
                        'optimizer_state_dict': optimizer.state_dict(),  # optimizer state
                        'epoch': epoch,  # save the current epoch
                        'best_val_loss': best_val_loss,  # the best validation loss
                        'counter': counter,  # early stopping counter
                        'splits': splits,  # training splits
                        'processed_splits': processed_splits,  # track which splits have already been processed
                        'last_split_idx': split_idx,  # the index of the last split
                        'last_session': s,  # the last session in the current split
                        'last_session_idx': session_idx,  # the index of the last session in the current split
                    }
                    torch.save(checkpoint, checkpoint_path)

                else:
                    counter += 1
                    if counter >= patience:
                        print("EARLY STOPPING - onto next split")
                        break

            # test on held-out session after all epochs complete
            with torch.no_grad():
                test_dist = model(X_test[:,:,0].unsqueeze(1), X_test[:,:,-1].unsqueeze(1))
                test_loss = NLL(test_dist, y_test).mean()
                print(f'Test session: S{s + 1}, Test Loss: {test_loss.item():.4f}')

        # mark current split as processed
        processed_splits.append(split_idx)
        print(f"Split {split_idx + 1} processed.")

    end_time = time.time()
    print("TRAINING COMPLETE: time ", (end_time - start_time) / 3600, " hours.")

    return


def main():

    lr = 5e-4
    batch_size = 128
    n_epochs = 500

    # local or remote training
    host = 'localhost'
    # host = '104.197.247.27'

    # extract filtered BVP signal from SQL database
    database = "smartwatch_raw_data_all"
    conn = psycopg2.connect(
        dbname=database,
        user="postgres",
        password="newpassword",
        host=host,
        port=5432
    )
    cur = conn.cursor()


    # extract unique datasets
    cur.execute("SELECT DISTINCT dataset FROM ma_filtered_data;")
    datasets = [row[0] for row in cur.fetchall()]

    train_model(cur, conn, datasets, batch_size, n_epochs, lr)

    cur.close()
    conn.close()


if __name__ == '__main__':
    main()

