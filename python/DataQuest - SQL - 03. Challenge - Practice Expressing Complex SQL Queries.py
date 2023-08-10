import sqlite3
jobs = sqlite3.connect('jobs.db')
c = jobs.cursor()

def print_sql(sql_command):
    for row in c.execute(sql_command):
        print(row)  

x='''select College_jobs, Median, Unemployment_rate 
from recent_grads 
limit 10;'''

print_sql(x)

x='''select major 
from recent_grads 
where Major_category='Arts' 
limit 5;'''

print_sql(x)

x='''select Major,Total,Median,Unemployment_rate 
from recent_grads 
where (Major_category != 'Engineering') 
and (Unemployment_rate > 0.065 or Median <= 50000)
limit 5;'''

print_sql(x)

x='''select major
from recent_grads
where Major_category!='Engineering'
order by major desc
limit 10;'''

print_sql(x)

