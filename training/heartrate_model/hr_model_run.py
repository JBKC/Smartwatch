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

def data_generator(cur, folds, test_idx, batch_size):
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

    def fetch_batch(sessions, offset, batch_size):
        '''
        Fetch data one batch at a time
        '''
        x_batch, y_batch = [], []

        for dataset, session in sessions:
            query_ppg = f"""
                SELECT ppg FROM ma_filtered_data WHERE dataset=%s AND session_number=%s 
                LIMIT %s OFFSET %s;
            """
            cur.execute(query_ppg, (dataset, session, batch_size, offset))
            ppg_windows = np.array([row[0] for row in cur.fetchall()])

            query_label = f"""
                SELECT label FROM ma_filtered_data WHERE dataset=%s AND session_number=%s 
                LIMIT %s OFFSET %s;
            """
            cur.execute(query_label, (dataset, session, batch_size, offset))
            labels = np.array([row[0] for row in cur.fetchall()])

            if ppg_windows.size == 0 or labels.size == 0:
                continue  # No more data for this session

            x_pairs, y = temporal_pairs(ppg_windows, labels)
            x_batch.append(x_pairs)
            y_batch.append(y)

        if not x_batch:
            raise StopIteration  # All data for this split is processed

        return np.concatenate(x_batch, axis=0), np.concatenate(y_batch, axis=0)

    offset = 0
    while True:
        try:
            x_train, y_train = fetch_batch(train_sessions, offset, batch_size)
            x_val, y_val = fetch_batch(val_sessions, offset, batch_size)
            x_test, y_test = fetch_batch(test_sessions, offset, batch_size)
            yield x_train, y_train, x_val, y_val, x_test, y_test
            offset += batch_size
        except StopIteration:
            break

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

    def save_checkpoint(state, filename):
        """ Save a checkpoint to a file """
        torch.save(state, filename)
    def load_checkpoint(filename):
        """ Load a checkpoint from a file """
        if os.path.exists(filename):
            return torch.load(filename)
        return None

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

    # iterate over folds (each fold trains a new model)
    for test_idx in range(len(folds)):

        # initialise model
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

        ### extract & pass batches through model one by one
        # generate data for single batch in given fold
        for x_train, y_train, x_val, y_val, x_test, y_test in data_generator(cur, folds, test_idx, batch_size):

            x_train, y_train = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
            x_val, y_val = torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
            x_test, y_test = torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

            print(f"Train: {x_train.shape}, {y_train.shape}")
            print(f"Validation: {x_val.shape}, {y_val.shape}")
            print(f"Test: {x_test.shape}, {y_test.shape}")

            # batch data
            train_dataset = TensorDataset(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            # begin training
            start_time = time.time()

            for epoch in range(n_epochs):
                model.train()
                epoch_loss = 0

                for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                    optimizer.zero_grad()

                    # prep data for model input - shape is (batch_size, n_channels, sequence_length) = (256, 1, 256)
                    x_cur = x_batch[:, :, 0].unsqueeze(1)
                    x_prev = x_batch[:, :, -1].unsqueeze(1)

                    # forward pass through model (convolutions + attention + probabilistic)
                    dist = model(x_cur, x_prev)

                    # calculate training loss on distribution
                    loss = NLL(dist, y_batch).mean()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                    print(f'Fold: {test_idx + 1}/{len(folds)}, Batch: [{batch_idx + 1}/{len(train_loader)}], '
                          f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {loss:.4f}')

                epoch_loss /= len(train_loader)
                if epoch_loss < best_train_loss:
                    best_train_loss = epoch_loss

                # validation on whole validation set after each epoch
                model.eval()

                with torch.no_grad():
                    val_dist = model(x_val[:,:,0].unsqueeze(1), x_val[:,:,-1].unsqueeze(1))
                    val_loss = NLL(val_dist, y_val).mean()          # average validation across all windows

                    print(f'Fold: {test_idx + 1}/{len(folds)}, Epoch [{epoch + 1}/{n_epochs}], '
                          f'Validation Loss: {val_loss.item():.4f}')

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

            # test on test data after all epochs complete for current fold
            with torch.no_grad():
                test_dist = model(x_test[:,:,0].unsqueeze(1), x_test[:,:,-1].unsqueeze(1))
                test_loss = NLL(test_dist, y_test).mean()

                print(f'Fold: {test_idx + 1}/{len(folds)}, Test Loss: {test_loss.item():.4f}')

            end_time = time.time()
            print(f"TRAINING COMPLETE - Fold: {test_idx + 1}/{len(folds)}, "
                  f"time: {(end_time - start_time) / 3600:.2f} hours.")

            # save down current best model state
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

