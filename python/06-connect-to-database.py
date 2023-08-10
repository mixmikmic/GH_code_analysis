import numpy as np
import pandas as pd

import psycopg2 as pg2

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().magic('matplotlib inline')

# Loading the data saved from the last notebook
X_train = np.load('./_data/X_train.npy')
y_train = np.load('./_data/y_train.npy')
X_val = np.load('./_data/X_val.npy')
y_val = np.load('./_data/y_val.npy')
X_test = np.load('./_data/X_test.npy')

# Connect to madelon database
con = pg2.connect(host='34.211.227.227',
                  dbname='postgres',
                  user='postgres')
cur = con.cursor()

# Print size (data) of madelon database
cur.execute("SELECT COUNT(*) FROM madelon;")
print(cur.fetchall())

# Print size (Mbytes) of madelon database
cur.execute("SELECT pg_size_pretty( pg_total_relation_size('madelon') );")
print(cur.fetchall())

query = "SELECT '48', '64', '105', '128', '241', '323', '336', '338', '378', '442', '453', '472', '475' FROM madelon ORDER BY Random() LIMIT 6500;"
query

cur.execute(query)
db_sample = cur.fetchall()

db_sample_df = pd.DataFrame(db_sample)

# # Download sample of dataset
# query = "SELECT * FROM madelon ORDER BY RANDOM() LIMIT 6500"
# cur.execute(query)
# db_sample = cur.fetchall()
# db_sample_df = pd.DataFrame(db_sample)

print(db_sample_df.shape)
print(db_sample_df.info())
db_sample_df.head()

db_sample_np = db_sample_df.as_matrix(columns=None)

db_sample_np

# Pickle np.array
np.save('./_data/db_sample_np', db_sample_np)

pickle_test = np.load('./_data/db_sample_np.npy')

pickle_test.shape

pickle_test[:,-1]



