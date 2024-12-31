'''
Training Step 2: Perform motion artifact removal on raw training data using linear adaptive filter
Saves down to new table within PostgreSQL database
'''

import psycopg2

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