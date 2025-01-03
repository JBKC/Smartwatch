'''
Script for pulling stats of training data
'''

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

def main():

    def get_activities():
        cur.execute("SELECT DISTINCT activity FROM session_data")
        activities = cur.fetchall()
        return [activity[0] for activity in activities]

    def activity_counter():
        '''
        Counts number of 8-second windows for each activity
        '''
        for act in acts:
            # query = (f"SELECT COUNT(*) FROM session_data "
            #          f"WHERE activity = {act} "
            #          f"AND dataset = 'ppg_dalia' "
            #          f"AND session_number = 'S7';")
            query = ("SELECT COUNT(*) FROM session_data "
                     "WHERE activity = %s ")
            cur.execute(query, (act,))
            count = cur.fetchone()[0]
            name = activity_mapping[act]
            print(f"{name} | {count}")

    def average_hr():
        '''
        Gets average heart rate for each activity
        '''
        # query = (f"SELECT activity, AVG(label::FLOAT) "
        #          f"AS average_heart_rate FROM session_data "
        #          f"WHERE dataset = 'ppg_dalia'"
        #          f"AND session_number = 'S7'"
        #          f"GROUP BY activity;")
        query = ("SELECT activity, AVG(label::FLOAT) "
                 "AS average_heart_rate FROM session_data "
                 "GROUP BY activity;")
        cur.execute(query)
        results = cur.fetchall()

        # Print results
        print("Activity | Average Heart Rate")
        for row in results:
            name = activity_mapping[row[0]]
            print(f"{name} | {row[1]:.2f}")


    database = "smartwatch_raw_data_all"

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