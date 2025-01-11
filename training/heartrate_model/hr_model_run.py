'''
Training Step 3: Main supervised attention-based model for extracting heartrate from filtered BVP
'''

import numpy as np
from sklearn.utils import shuffle
import time
import psycopg2
import os
from hr_model import TemporalAttentionModel
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import psutil
from wandb_logger import WandBLogger
import datetime
import copy
import subprocess



def data_generator(cur, folds, test_idx):
    '''
    Extracts data from SQL table, divides into LOSO train, val and test splits, and creates temporal pairs
    :param folds: list of sessions divided into folds
    :param test_idx: index of the current test session
    :return: all data and labels for train, val and test splits
    '''

    # extract training sessions (!= test session)
    train_sessions = [(dataset, session) for i, group in enumerate(folds) if i!=test_idx
                      for dataset, session in group]
    test_sessions = folds[test_idx]

    n_val = int(len(test_sessions) * 4/5)           # number of sessions to retain in validation set
    val_sessions = test_sessions[:n_val]
    test_sessions = test_sessions[n_val:]

    # print(len(train_sessions), len(val_sessions), len(test_sessions))

    # get temporal pairings for training data
    x_train, y_train = [], []
    x_val, y_val = [], []
    x_test, y_test = [], []

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

    # extract temporal pairs for validation data
    for dataset, session in val_sessions:

        query = f"SELECT ppg FROM ma_filtered_data WHERE dataset=%s AND session_number=%s;"
        cur.execute(query, (dataset,session))
        ppg_windows = np.array([row[0] for row in cur.fetchall()])

        query = f"SELECT label FROM ma_filtered_data WHERE dataset=%s AND session_number=%s;"
        cur.execute(query, (dataset,session))
        labels = np.array([row[0] for row in cur.fetchall()])

        x_pairs, y = temporal_pairs(ppg_windows, labels)

        x_val.append(x_pairs)
        y_val.append(y)

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
    x_val = np.concatenate(x_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    yield x_train, y_train, x_val, y_val, x_test, y_test

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

def train_model(cur, conn, datasets, batch_size, n_epochs, lr):

    def torch_convert(X, y, batch_size=10):
        '''
        Necessary when converting large amounts of numpy data to torch data in one go
        '''

        x_batches = []
        y_batches = []

        # Process in batches
        for i in range(0, len(X), batch_size):
            x_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # Convert each batch to PyTorch tensor
            x_batches.append(torch.tensor(x_batch, dtype=torch.float32))
            y_batches.append(torch.tensor(y_batch, dtype=torch.float32))

        # Concatenate all batches
        X_tensor = torch.cat(x_batches, dim=0)
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

    def create_folds(datasets,n_folds):
        '''
        Create grouped LOSO splits for training
        :return: list of folds
        '''

        all_sessions = []

        for dataset in datasets:
            query = f"SELECT DISTINCT session_number FROM ma_filtered_data WHERE dataset=%s;"
            cur.execute(query, (dataset,))
            sessions = [row[0] for row in cur.fetchall()]
            all_sessions.extend([(dataset, session) for session in sessions])

        np.random.shuffle(all_sessions)
        folds = np.array_split(all_sessions, n_folds)

        return folds


    # checkpoint_path = f'saved_models/hr_model/.pth'
    # checkpoint = torch.load(checkpoint_path)
    # if checkpoint:
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     last_split_idx = checkpoint['last_split_idx']
    #     last_session_idx = checkpoint['last_session_idx']
    #     processed_splits = checkpoint['processed_splits']
    #     best_val_loss = checkpoint['best_val_loss']
    #     print(f"Resuming from split {last_split_idx + 1}, session {last_session_idx + 1}")
    # else:
    #     last_split_idx = -1
    #     last_session_idx = -1
    #     processed_splits = []
    #     best_val_loss = float('inf')
    #     print("Starting training from scratch")

    # predefine the folds for LOSO split
    folds = create_folds(datasets,n_folds=5)

    print(f"Memory usage: {psutil.virtual_memory().percent}%")

    # iterate over folds
    for test_idx in range(len(folds)):

        # initialise model (each fold trains a new model)
        model = TemporalAttentionModel()
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)

        print(f"LOSO fold {test_idx + 1} of {len(folds)}")

        # early stopping parameters
        patience = 10
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        best_model_state = None
        counter = 0

        # create separate logs for each fold / model run
        logger = WandBLogger(
            project_name="smartwatch-hr-estimator",
            config={
                'folds': folds,
                'fold_idx': test_idx,
                "learning_rate": lr,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "model_architecture": "AdaptiveLinearModel"
            }
        )

        # generate all data for given fold
        for x_train, y_train, x_val, y_val, x_test, y_test in data_generator(cur, folds, test_idx):

            # shut down Docker for memory reasons
            cur.close()
            conn.close()
            subprocess.run(["docker", "stop", "timescaledb"], check=True)
            print(f"Docker ended")

            # globally shuffle training data
            perm_train = np.random.permutation(x_train.shape[0])
            x_train = x_train[perm_train]
            y_train = y_train[perm_train]

            x_train, y_train = torch_convert(x_train, y_train)
            x_val, y_val = torch_convert(x_val, y_val)
            x_test, y_test = torch_convert(x_test, y_test)

            print(f"Train: {x_train.shape}, {y_train.shape}")
            print(f"Validation: {x_val.shape}, {y_val.shape}")
            print(f"Test: {x_test.shape}, {y_test.shape}")

            ## chunk data for smooth model loading
            chunk_size = batch_size * 10
            n_chunks = (len(x_train) + chunk_size - 1) // chunk_size
            n_batches = (len(x_train) + batch_size - 1) // batch_size

            start_time = time.time()

            ### training loop ###
            for epoch in range(n_epochs):
                model.train()
                epoch_loss = 0

                # chunk training data before creating batches
                for chunk_idx in range(n_chunks):

                    start_idx = chunk_idx * chunk_size
                    end_idx = min((chunk_idx + 1) * chunk_size, len(x_train))
                    x_train_chunk = x_train[start_idx:end_idx]
                    y_train_chunk = y_train[start_idx:end_idx]

                    # print(x_train_chunk.shape, y_train_chunk.shape)

                    # create batches within current chunk
                    train_dataset = TensorDataset(x_train_chunk, y_train_chunk)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

                    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                        optimizer.zero_grad()

                        # print(x_batch.shape, y_batch.shape)

                        # get into format for model input - shape is (batch_size, 1, 256)
                        x_cur = x_batch[:, :, :, 0]
                        x_prev = x_batch[:, :, :, -1]
                        # print(f"Model inputs (x_cur, x_prev): {x_cur.shape}, {x_prev.shape}")
                        # print(f"Memory usage: {psutil.virtual_memory().percent}%")

                        # forward pass through model (convolutions + attention + probabilistic)
                        dist = model(x_cur, x_prev)

                        # calculate training loss on distribution
                        loss = NLL(dist, y_batch).mean()
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()

                        print(f'Fold: {test_idx + 1}/{len(folds)}, '
                              f'Epoch [{epoch + 1}/{n_epochs}], ' 
                              f'Chunk {chunk_idx + 1}/{n_chunks}, '
                              f'Batch: {batch_idx + 1}/{len(train_loader)}, '
                              f'Train Loss: {loss:.4f}')

                        del x_batch, y_batch

                    # save memory
                    del train_loader, x_train_chunk, y_train_chunk

                # overall epoch loss
                epoch_loss /= n_batches
                if epoch_loss < best_train_loss:
                    best_train_loss = epoch_loss

                ### batched validation on whole set after each epoch ###
                model.eval()
                val_dataset = TensorDataset(x_val, y_val)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                val_loss = 0

                with torch.no_grad():
                    for batch_idx, (x_batch, y_batch) in enumerate(val_loader):

                        # Prepare the data for model input
                        x_cur = x_batch[:, :, :, 0]
                        x_prev = x_batch[:, :, :, -1]

                        # Pass the chunk through the model
                        dist = model(x_cur, x_prev)

                        loss = NLL(dist, y_batch).mean()
                        val_loss += loss.item()

                        del x_batch, y_batch

                val_loss /= len(val_loader)
                del val_loader

                print(f'Fold: {test_idx + 1}/{len(folds)}, Epoch [{epoch + 1}/{n_epochs}], '
                      f'Validation Loss: {val_loss:.4f}')

                # upload metrics to logger for each epoch
                metrics = {"train loss": epoch_loss, "validation loss": val_loss}
                metrics.update({f'grad_{name}': param.grad.norm().item()
                                for name, param in model.named_parameters() if param.grad is not None})
                logger.log_metrics(metrics, step=epoch)


                # early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    counter = 0

                else:
                    counter += 1

                if counter >= patience:
                    print("EARLY STOPPING - onto next fold")
                    break

                # save down current best model state (trial mode - saves after every epoch)
                checkpoint = {
                    'folds': folds,
                    'fold_idx': test_idx,
                    'model_state_dict': best_model_state,  # model weights
                    'optimizer_state_dict': optimizer.state_dict(),  # optimizer state
                    'best_train_loss': best_train_loss,
                    'best_val_loss': best_val_loss,  # the best validation loss
                    'counter': counter,  # early stopping counter
                }
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                model_path = f"saved_models/hr_model/hr_model_{timestamp}.pth"
                torch.save(checkpoint, model_path)

            ### run batches of test data ###
            model.eval()
            test_dataset = TensorDataset(x_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loss = 0

            with torch.no_grad():
                for batch_idx, (x_batch, y_batch) in enumerate(test_loader):

                    # Prepare the data for model input
                    x_cur = x_batch[:, :, :, 0]
                    x_prev = x_batch[:, :, :, -1]

                    # Pass the chunk through the model
                    dist = model(x_cur, x_prev)

                    loss = NLL(dist, y_batch).mean()
                    test_loss += loss.item()

                    del x_batch, y_batch

            test_loss /= len(test_loader)
            del test_loader

            print(f'Fold: {test_idx + 1}/{len(folds)}, Test Loss: {test_loss.item():.4f}')

            end_time = time.time()
            print(f"TRAINING COMPLETE - Fold: {test_idx + 1}/{len(folds)}, "
                  f"time: {(end_time - start_time) / 3600:.2f} hours.")

    return


def main():

    lr = 5e-4
    batch_size = 128
    n_epochs = 500

    # local or remote training
    host = 'localhost'
    # host = '104.197.247.27'

    # start Docker container
    subprocess.run(["docker", "start", "timescaledb"], check=True)
    print(f"Docker started")
    time.sleep(2)

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


if __name__ == '__main__':
    main()