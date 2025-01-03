'''
Script for pulling stats of training data
'''

import psycopg2

activity_mapping = {
    0: "sitting still",
    1: "stairs",
    2: "table football",
    3: "cycling",
    4: "driving",
    5: "lunch break",
    6: "walking",
    7: "working at desk",
    8: "running"
}

def main():

    def get_activities():
        cur.execute(f"SELECT DISTINCT activity FROM {table}")
        activities = cur.fetchall()
        return [activity[0] for activity in activities]

    def activity_counter():
        '''
        Counts number of 8-second windows for each activity
        '''
        for act in acts:
            query = f"SELECT COUNT(*) FROM {table} WHERE activity = {act};"
            cur.execute(query)
            count = cur.fetchone()[0]
            name = activity_mapping[act]
            print(f"{name} | {count}")

    def average_hr():
        '''
        Gets average heart rate for each activity
        '''
        query = """SELECT activity, AVG(label) AS average_heart_rate FROM session_data GROUP BY activity;"""
        cur.execute(query)
        results = cur.fetchall()

        # Print results
        print("Activity | Average Heart Rate")
        for row in results:
            print(f"{row[0]} | {row[1]:.2f}")



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

    acts = get_activities()
    acts.sort()

    activity_counter()
    # average_hr()

    cur.close()
    conn.close()


if __name__ == '__main__':
    main()