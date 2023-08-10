import sys,os,os.path
os.environ['IBM_DB_HOME']='C:\Program Files\IBM\SQLLIB'
get_ipython().system('pip install ipython-sql')
get_ipython().system('pip install ibm_db ')
get_ipython().system('pip install ibm_db_sa')

#import ibm_db
#import ibm_db_sa
import sqlalchemy
get_ipython().magic('load_ext sql')
get_ipython().magic('run db2.ipynb')

user='db2admin'
host='localhost'
# Define filename for passwords
filename = 'ember_variables.py'
# source the file
get_ipython().magic('run $filename')
password = LocalDB2password

get_ipython().magic('sql db2+ibm_db://$user:$password@$host:50000/SAMPLE')



