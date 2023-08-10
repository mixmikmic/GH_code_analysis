import pandas as pd
from sqlalchemy import create_engine # database connection
import datetime as dt
from sqlalchemy.orm import sessionmaker

path = r'C:\Users\angelddaz\OneDrive\Documents\data_training\data\RawDelData.csv'
disk_engine = create_engine('sqlite:///data.db') # Initializes database with filename 'data.db' in current directory
connection = disk_engine.connect()

j = 0
index_start = 1

for df in pd.read_csv(path, iterator=True, encoding='utf-8'):
    df['Timestamp'] = pd.to_datetime(df['Timestamp']) # Convert to datetimes, welp this line didn't work haha #TOFIX
    df.index += index_start
    columns = ['Key', 'Date', 'mmdd', 'DayOfTheWeek', 'Area', 'Distance', 'Timestamp', 'Tip', 'OrderAmount', 'TipPercent',                'Housing', 'GenderOfTipper', 'Cash/Credit_Tip', 'Cash/Credit_Order', 'PersonWhoDelivered', 'Area(text)',                'Latitude', 'Longitude', 'month']
    j+=1
    df.to_sql('data', disk_engine, if_exists='replace')
    index_start = df.index[-1] + 1
    
connection.close()

#Some practice queries to see if our database creation above works
df = pd.read_sql_query('SELECT Key, Tip, Housing FROM data WHERE Housing = "Apartment" ', disk_engine)
print df.tail(), "\n"
print pd.read_sql_query('SELECT Count(*) FROM data', disk_engine)

from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://maxcdn.icons8.com/Share/icon/Data//database1600.png", width=300, height=300)
#AWESOME. I have the database functional and ready.

#1. Find out how many deliveries Angel took
print "#1"
print pd.read_sql_query('SELECT Count(*) AS [Angel Dels] FROM data AS d1                        WHERE d1.PersonWhoDelivered = "Angel"', disk_engine), "\n"

#2. Find out how many deliveries Sammie took
print "#2"
print pd.read_sql_query('SELECT Count(*) AS [Sam Dels] FROM data AS d1                        WHERE d1.PersonWhoDelivered = "Sammie"', disk_engine), "\n"

#print pd.read_sql_query('SELECT * FROM data', disk_engine)

#I'm going to try using sqlite using the database I created
import sqlite3
db = sqlite3.connect('data.db')

cursor = db.cursor()
cursor.execute( '''SELECT Key, PersonWhoDelivered, Tip FROM data WHERE Housing ="Apartment" 
                AND Tip >= 10 ORDER BY PersonWhoDelivered, Tip''')
print cursor.fetchall()
db.close()



