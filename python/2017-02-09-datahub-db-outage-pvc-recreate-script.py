import pandas as pd
import numpy as np

get_ipython().system('head pvcs.csv')

df = pd.DataFrame.from_csv('pvcs.csv')
df.head()

df['username'] = np.array(df.index.str.extract('\w+-([\w-]+)-\d+$'))
df['id'] = np.array(df.index.str.extract('\w+-\w+-(\d+)$'))
df.head()

df.tail()

valids = df[~df['AGE'].str.contains('m')].dropna()
valids = valids[valids['id'] != '14']
valids.head()

valids['username'].head().value_counts()

valids['id'].value_counts().head()

len(valids['id'])

import sqlite3
conn = sqlite3.connect('jupyterhub.sqlite')
c = conn.cursor()

c.execute('PRAGMA TABLE_INFO({})'.format('users'))

c.fetchall()

c.execute('SELECT * FROM users LIMIT 20')
current_users = c.fetchall()
current_users[:2]

admins = {user[1] for user in current_users if user[3] == 1}
admins

# I just did this manually in the sqlite3 CLI
# c.execute('DROP FROM users where id > -1')
# c.fetchall()

import datetime
import itertools

records = list(zip(
    valids['id'].astype(int),
    valids['username'],
    itertools.repeat('NULL'),
    [1 if name in admins else 0 for name in valids['username']],
#     itertools.repeat('NULL'), # Pick a random valid value
    itertools.repeat('2017-02-09 09:07:03.936620'), # Pick a random valid value
    itertools.repeat('thisisadummycookiehopefullyitworks'),
    itertools.repeat('NULL'),
    itertools.repeat('NULL')
))
records[:3]

# I hate everything
def record_to_sql(record):
    return "INSERT INTO users VALUES({}, '{}', {}, '{}', '{}', '{}', {}, {})".format(
        *record
    )

sql_statements = [record_to_sql(record) for record in records]
record_to_sql(records[0])

with conn:
    c = conn.cursor()
    for statement in sql_statements:
        c.execute(statement)

c.execute('SELECT * FROM users LIMIT 20')
c.fetchall()



