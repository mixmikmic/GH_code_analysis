# NOTE: You have to use the m2 database, 
# the 'factbook' from old missions is different

import sqlite3
conn = sqlite3.connect('factbook_m2.db')

q='''ALTER TABLE facts
ADD leader text;'''
conn.execute(q)

q = '''CREATE TABLE states(
    id integer PRIMARY KEY,
    name text,
    area integer,
    country integer,
    FOREIGN KEY(country) REFERENCES facts(id)
);'''

conn.execute(q)

#create fetchall function
def fetchall(q_x):
    names = []
    [names.append(name[0]) for name in conn.execute(q_x).description]
    print(names,'\n')
    for row in conn.execute(q_x).fetchall():
        print(row)

q = '''SELECT * from landmarks
INNER JOIN facts
ON landmarks.country == facts.id;'''

fetchall(q)

q='''SELECT * from landmarks
LEFT OUTER JOIN facts
ON landmarks.country == facts.id;'''

fetchall(q)

