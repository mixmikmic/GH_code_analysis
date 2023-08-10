from impala.dbapi import connect
conn = connect(host='impalasrv-prod', port=21050, kerberos_service_name='impala', auth_mechanism='GSSAPI')

cur = conn.cursor()

cur.execute('select * from test2.emp limit 2')

cur.fetchall()

cur = conn.cursor()

cur.execute('select * from test2.emp')

from impala.util import as_pandas
df = as_pandas(cur)

df.head()

cur = conn.cursor()
cur.execute('select ename, sal from test2.emp')
df = as_pandas(cur)

get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.style.use('ggplot')

df.plot()



