# Load required libraries
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
from sklearn.externals import joblib

# Set Postgres credentials
db_name1 = 'section1_db'
db_name2 = 'section2_db'
usernm = 'redwan'
host = 'localhost'
port = '5432'
# pwd = ''

# Create engines for both databases
engine1 = create_engine(
    'postgresql://{}:{}@{}:{}/{}'.format(usernm, pwd, host, port, db_name1)
)

engine2 = create_engine(
    'postgresql://{}:{}@{}:{}/{}'.format(usernm, pwd, host, port, db_name2)
)

# Create a new database for each section if it already does not exist
if not database_exists(engine1.url):
    create_database(engine1.url)

if not database_exists(engine2.url):
    create_database(engine2.url)

# Display whether the database exists
print(database_exists(engine1.url), database_exists(engine2.url) )

# Load DataFrames from pickle files
section1_df = joblib.load(
    'data/extracted_data/section1_all_features_20000-24558.pkl'
)

section2_df = joblib.load(
    'data/extracted_data/section2_all_features_20000-24558.pkl'
)

# Append data into the corresponding SQL database
section1_df.to_sql(
    name=db_name1, 
    con=engine1,
    if_exists='append'
)

section2_df.to_sql(
    name=db_name2, 
    con=engine2,
    if_exists='append'
)

# Connect to a database
con1 = psycopg2.connect(
    database=db_name1, 
    host='localhost',
    user=usernm,
    password=pwd
)

# Define a SQL query for loading a campaign section
sql_query = """
SELECT * 
  FROM section1_db;
"""

# Perform SQL query and store results in a DataFrame
test_data_from_sql = pd.read_sql_query(sql_query, con1)

# Display the first five rows
test_data_from_sql.tail()

# Display the number of entries in the database
len(test_data_from_sql)

# Display DataFrame information
test_data_from_sql.info()

