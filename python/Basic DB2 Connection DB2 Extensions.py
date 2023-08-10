#These are still required with the Db2 Extensions if it is your first time running
# Db2 and Jupyter Notebook. These do not have to be run for subsequent executions.
import sys,os,os.path
os.environ['IBM_DB_HOME']='C:\Program Files\IBM\SQLLIB'
get_ipython().system('pip install ipython-sql')
get_ipython().system('pip install ibm_db ')
get_ipython().system('pip install ibm_db_sa')

#Modules required without Db2 extensions are commented out below
#import ibm_db
#import ibm_db_sa
#import sqlalchemy
#%load_ext sql
#With Db2 extensions, the follwoing line is all that is needed
get_ipython().magic('run db2.ipynb')

user='db2admin'
host='localhost'
# Define filename for passwords
filename = 'ember_variables.py'
# source the file
get_ipython().magic('run $filename')
password = LocalDB2password

#Connection string used without db2 extensions is commented out below
#%sql db2+ibm_db://$user:$password@$host:50000/SAMPLE
#Connection string that only works with db2 extensions
get_ipython().magic('sql CONNECT TO SAMPLE USER $user USING $password HOST $host PORT 50000')

get_ipython().magic('sql CONNECT TO SAMPLE USER $user USING ? HOST $host PORT 50000')



