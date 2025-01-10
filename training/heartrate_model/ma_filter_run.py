'''
Training Step 2: Perform motion artifact removal on raw training data using linear adaptive filter
Saves filtered data to new table within PostgreSQL database
Saves individual model parameter sets for each activity
'''

import numpy as np
import os
import matplotlib.pyplot as plt
from ma_filter import AdaptiveLinearModel
from wandb_logger import WandBLogger
import torch
import torch.optim as optim
import psycopg2
import datetime
import copy

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

def train_ma_filter(cur, conn, acts, batch_size, n_epochs, lr, select=None):
    '''
    Uses ma_filter architecture to remove motion artifacts from raw PPG signal by activity
    Optionality to train on single selected activity or all
    '''

    # check environment (Google Colab or local)
    try:
        import google.colab
        COLAB = True
        from google.colab import files
    except ImportError:
        COLAB = False

    # save directory for models
    if not os.path.exists("saved_models/ma_filter/"):
        os.makedirs("saved_models/ma_filter/")

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
    if select is not None:
        if select in acts:
            acts = [select]

    for act in acts:

        # initialise model logger (separate run for each activity)
        logger = WandBLogger(
            project_name="smartwatch-adaptive-linear-filter",
            config={
                "activity": activity_mapping[act],
                "learning_rate": lr,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "model_architecture": "AdaptiveLinearModel"
            }
        )

        # initialise activity-specific filter model
        model = AdaptiveLinearModel()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=1e-2)

        print(f"Training filter for {activity_mapping[act]}...")

        # early stopping parameters
        patience = 10
        best_loss = float('inf')
        best_model_state = None
        counter = 0

        # training loop
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0          # loss over all batches per epoch

            # generate batches within each activity
            for batch in fetch_activity_data(act, batch_size):
                ppg = np.expand_dims(np.array([row['ppg'] for row in batch]), axis=1)
                acc = np.array([row['acc'] for row in batch])
                # print(ppg.shape)                # (batch_size, 1, 256)
                # print(acc.shape)                # (batch_size, 3, 256)

                # concatenate ppg + accelerometer signal data -> (n_windows, 4, 256)
                X = np.concatenate((ppg, acc), axis=1)
                # print(X.shape)                  # (batch_size, 4, 256)
                X, ms, stds = z_normalise(X)                    # batch Z-normalisation
                X = torch.from_numpy(np.expand_dims(X, axis=1)).float()

                # accelerometer data == inputs:
                x_acc = X[:, :, 1:, :]                 # (batch_size, 1, 3, 256)
                # PPG data == targets:
                x_ppg = X[:, :, :1, :]                 # (batch_size, 1, 1, 256)

                # plt.plot(x_ppg[0,0,0,:])
                # plt.show()

                # print(x_acc.shape)
                # print(x_ppg.shape)

                # forward pass through CNN to get x_ma (motion artifact estimate)
                x_ma = model(x_acc)
                loss = model.adaptive_loss(y_true=x_ppg, y_pred=x_ma)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # print loss
                print(f'Activity: {activity_mapping[act]}, Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss:.4f}')

            # log metrics
            metrics = {"loss": epoch_loss}
            metrics.update({f'grad_{name}': param.grad.norm().item()
                            for name, param in model.named_parameters() if param.grad is not None})
            logger.log_metrics(metrics, step=epoch)

            # create copy of model state for early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_state = copy.deepcopy(model.state_dict())
                counter = 0

            else:
                counter += 1

            # early stopping
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} for {activity_mapping[act]}.")
                break

        #### training complete for given activity ####
        print(f"Training complete for {activity_mapping[act]}...")

        # save best model state
        model.load_state_dict(best_model_state)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_path = f"saved_models/ma_filter/{activity_mapping[act]}_{timestamp}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved locally at {model_path}")

        # save on Drive
        if COLAB:
            try:
                files.download(model_path)
                print(f"Model saved on Drive")

            except Exception as e:
                print(f"Failed to download model: {e}")

        #### perform filtering on trained model for given activity ####

        # pass signal through trained model to filter by batch
        for batch in fetch_activity_data(act, batch_size):
            ppg = np.expand_dims(np.array([row['ppg'] for row in batch]), axis=1)
            acc = np.array([row['acc'] for row in batch])

            # Concatenate and normalize
            X = np.concatenate((ppg, acc), axis=1)  # Shape: (batch_size, 4, 256)
            X, ms, stds = z_normalise(X)           # Batch Z-normalization
            X = torch.from_numpy(np.expand_dims(X, axis=1)).float()

            x_ppg = X[:, :, :1, :]  # (batch_size, 1, 1, 256)
            x_acc = X[:, :, 1:, :]  # (batch_size, 1, 3, 256)

            with torch.no_grad():
                x_bvp = x_ppg[:, 0, 0, :] - model(x_acc)        # subtract motion artifact from raw signal to extract cleaned BVP

            # get signal into original shape (n_windows, 1, 256) and de-normalise
            x_bvp = torch.unsqueeze(x_bvp, dim=1).numpy()
            x_bvp = undo_normalisation(x_bvp, ms, stds)
            x_bvp = np.expand_dims(x_bvp[:,0,:], axis=1)            # keep only BVP (remove ACC)

            print(f"Processed batch: {len(batch)}, Filtered signal shape: {x_bvp.shape}")

            # Prepare rows for SQL insertions into new table
            rows_to_insert = [
                (
                    row['dataset'],  # Dataset name
                    row['session_number'],  # Session number
                    x_bvp[i].tolist(),  # Filtered PPG signal
                    row['acc'],  # Raw accelerometer data
                    row['activity'],  # Activity ID
                    row['label']  # Ground truth label
                )
                for i, row in enumerate(batch)
            ]

            # Insert filtered data into the new table
            query = """
                    INSERT INTO ma_filtered_data (dataset, session_number, ppg, acc, activity, label)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
            cur.executemany(query, rows_to_insert)
            conn.commit()  # Commit changes

        print(f"Filtered data saved for {activity_mapping[act]}.")

    return


def main():

    # model config
    lr = 1e-7
    batch_size = 256
    n_epochs = 1000

    # local or remote training
    host = 'localhost'
    # host = '104.197.247.27'

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
        host=host,
        port=5432
    )
    cur = conn.cursor()

    # get unique activities
    acts = get_activities(cur)

    # prompt to select which activity to train
    select = input("\nEnter activity ID to train (or press Enter to train ALL): ")
    if select.strip():
        try:
            select = int(select)
        except ValueError:
            print(f"Invalid input '{select}', not a number. Defaulting to ALL activities.")
            select = None

    # train filters & filter data
    train_ma_filter(cur, conn, acts, batch_size, n_epochs, lr, select)

    cur.close()
    conn.close()

if __name__ == '__main__':
    main()