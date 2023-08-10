import sys,os,os.path
os.environ['IBM_DB_HOME']='C:\Program Files\IBM\SQLLIB'
get_ipython().system('pip install ipython-sql')
get_ipython().system('pip install ibm_db ')
get_ipython().system('pip install ibm_db_sa')

import ibm_db
import ibm_db_sa
import sqlalchemy
get_ipython().magic('load_ext sql')

user='db2admin'
host='localhost'
# Define filename for passwords
filename = 'ember_variables.py'
# source the file
get_ipython().magic('run $filename')
password = LocalDB2password
db='SAMPLE'

get_ipython().magic('sql db2+ibm_db://$user:$password@$host:50000/$db')

inst_memory = get_ipython().magic('sql select memory_set_type     , db_name     , sum(memory_set_used)/1024 as used_mb from table(mon_get_memory_set(NULL,NULL,-2)) group by memory_set_type, db_name')
inst_memory

get_ipython().magic('matplotlib inline')
inst_memory.pie()

db_memory = get_ipython().magic('sql select memory_set_type     , memory_pool_type     , sum(memory_pool_used)/1024 as used_mb from table(mon_get_memory_pool(NULL,:db,-2)) where db_name=:db group by memory_set_type, memory_pool_type')
db_memory

get_ipython().magic('matplotlib inline')
db_memory.pie()



