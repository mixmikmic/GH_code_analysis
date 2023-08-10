import psycopg2
import pandas
import pandas.io.sql as psql
import ConfigParser
import os

USER_CRED_FILE = os.path.join(os.path.expanduser('~'), '.gpdbsandbox_user.cred')

def fetchDBCredentials(dbcred_file=USER_CRED_FILE):
    """
       Read database access credentials from the file in $HOME/.ipynb_dslab.cred
    """
    #Read database credentials from user supplied file
    conf = ConfigParser.ConfigParser()
    conf.read(dbcred_file)
    #host, port, user, database, password
    host = conf.get('database_creds','host')
    port = conf.get('database_creds','port')
    user = conf.get('database_creds','user')
    database = conf.get('database_creds','database')
    password = conf.get('database_creds','password')

    #Initialize connection string
    conn_str =  """dbname='{database}' user='{user}' host='{host}' port='{port}' password='{password}'""".format(                       
                    database=database,
                    host=host,
                    port=port,
                    user=user,
                    password=password
            )
    return conn_str

conn = psycopg2.connect(fetchDBCredentials())
df = psql.read_sql("""select random() as x, random() as y from generate_series(1, 10) q;""", conn)
df.head()

