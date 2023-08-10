import unicodecsv
def read_csv(filename):
    with open(filename, 'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)
    
daily_engagement = read_csv('daily_engagement.csv')

def get_unique_students(data):
    unique_students = set()
    for data_point in data:
        unique_students.add(data_point['acct'])
    return unique_students

unique_engagement_students = get_unique_students(daily_engagement)
len(unique_engagement_students)

import pandas as pd

daily_engagement = pd.read_csv('daily_engagement.csv')

len(daily_engagement['acct'].unique())



