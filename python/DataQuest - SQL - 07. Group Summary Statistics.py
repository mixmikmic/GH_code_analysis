#first import mikes custom function woooooooot!!!
#create fetchall function
import sqlite3
conn = sqlite3.connect('jobs.db') #set db conncetion
def fetchall(q_x):
    names = [] #initialize list of col names
    [names.append(name[0]) for name in conn.execute(q_x).description] #extract column names with description method
    print(names,'\n')
    for row in conn.execute(q_x).fetchall():
        print(row)

fetchall('Select * from recent_grads')

import sqlite3 as sql

conn = sql.connect('jobs.db')
q = 'SELECT * from recent_grads LIMIT 5;'
data = conn.execute(q).fetchall()

data

conn.execute('PRAGMA TABLE_INFO(recent_grads);').fetchall()

sql = """SELECT
        Major_category, Major, SUM(Employed), employed
        FROM recent_grads
        GROUP BY Major_category"""

fetchall(sql)

sql = """SELECT COUNT(DISTINCT Major_category) FROM recent_grads"""
fetchall(sql)

sql='''SELECT Major_category, AVG(ShareWomen) 
FROM recent_grads 
GROUP BY Major_category;'''

fetchall(sql)

sql = '''SELECT SUM(Men) AS total_men, SUM(Women) AS total_women 
FROM recent_grads;'''

fetchall(sql)

q = '''SELECT Major_category, AVG(Employed) / AVG(Total) AS share_employed 
FROM recent_grads 
GROUP BY Major_category
ORDER BY Major_category DESC;'''

conn.execute(q).fetchall()

sql = """SELECT 
        Major_category, 
        ROUND(AVG(ShareWomen),2) AS Share_Women
        FROM recent_grads 
        GROUP BY Major_category
        HAVING Share_Women > 0.5
        ORDER BY Share_Women DESC"""
fetchall(sql)

q = '''SELECT Major_category, AVG(Low_wage_jobs) / AVG(Total) AS share_low_wage 
FROM recent_grads 
GROUP BY Major_category 
HAVING share_low_wage > .1;'''

fetchall(q)

q = '''SELECT Major_category, ROUND(AVG(Low_wage_jobs) / AVG(Total),3) AS share_low_wage 
FROM recent_grads 
GROUP BY Major_category 
HAVING share_low_wage > .1;'''

fetchall(q)

sql='''SELECT ROUND(ShareWomen, 2) AS Share_Women, Major_category 
FROM recent_grads 
LIMIT 10;'''

fetchall(sql)

q='''SELECT 
    Major_category, 
    ROUND(AVG(College_jobs) / AVG(Total), 3) AS share_degree_jobs 
    FROM recent_grads 
    GROUP BY Major_category 
    HAVING share_degree_jobs < .3;'''

fetchall(q)

