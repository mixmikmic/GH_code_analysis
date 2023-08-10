import psycopg2 as pg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm

BATCH_SIZE = 500
SAMPLE_SIZE = 9200
SAMPLE_PERCENT = 5

con = pg2.connect(host='34.211.227.227',dbname='postgres', user='postgres')
cur = con.cursor(cursor_factory=RealDictCursor, name='database_cursor')
cur.execute('SELECT * FROM madelon TABLESAMPLE SYSTEM ({});'.format(SAMPLE_PERCENT))

database = []

pbar = tqdm(total=SAMPLE_SIZE//BATCH_SIZE)
while True:
    records = cur.fetchmany(size=BATCH_SIZE)

    if not records:
        break

    database += records
    pbar.update(1)

cur.close() 
con.close()
pbar.close()

database_1 = pd.DataFrame(database)

BATCH_SIZE = 500
SAMPLE_SIZE = 9200
SAMPLE_PERCENT = 4.5

con = pg2.connect(host='34.211.227.227',dbname='postgres', user='postgres')
cur = con.cursor(cursor_factory=RealDictCursor, name='database_2_cursor')
cur.execute('SELECT * FROM madelon TABLESAMPLE SYSTEM ({});'.format(SAMPLE_PERCENT))

database_2 = []

pbar = tqdm(total=SAMPLE_SIZE//BATCH_SIZE)
while True:
    records = cur.fetchmany(size=BATCH_SIZE)

    if not records:
        break

    database_2 += records
    pbar.update(1)

cur.close() 
con.close()
pbar.close()

database_2 = pd.DataFrame(database_2)

BATCH_SIZE = 500
SAMPLE_SIZE = 9200
SAMPLE_PERCENT = 5

con = pg2.connect(host='34.211.227.227',dbname='postgres', user='postgres')
cur = con.cursor(cursor_factory=RealDictCursor, name='database_3_cursor')
cur.execute('SELECT * FROM madelon TABLESAMPLE SYSTEM ({});'.format(SAMPLE_PERCENT))

database_3 = []

pbar = tqdm(total=SAMPLE_SIZE//BATCH_SIZE)
while True:
    records = cur.fetchmany(size=BATCH_SIZE)

    if not records:
        break

    database_3 += records
    pbar.update(1)

cur.close() 
con.close()
pbar.close()

database_3 = pd.DataFrame(database_3)

database_1.to_pickle('./Datasets/database_1.p')
database_2.to_pickle('./Datasets/database_2.p')
database_3.to_pickle('./Datasets/database_3.p')

