import psycopg2 # for interacting with PostgreSQL database
from FVSoutput_SQL_createDBtables import * # SQL query strings for creating FVS output tables

# We'll use the following helper function to write tables into the FVS Output database.
def create_tables(conn_str, table_SQLs, verbose=False):
    '''
    Creates tables in a PostgreSQL database to hold FVS Outputs.
    ===========
    Arguments:
    conn_str = A string for the database connection. 
    table_SQLs = A list of valid FVS table names (imported from FVSoutput_SQL_createDBtables.py).
    verbose = Will print out the SQL query strings used if set to True (defaults to False).
    '''
    with psycopg2.connect(conn_string) as conn:
        with conn.cursor() as cur:
            for SQL in table_SQLs:
                if verbose:
                    print(SQL)
                cur.execute(SQL)
                print('Created', SQL.split(' ')[2], end='... ')
    conn.close()
    print('Done.')

my_tables = [fvs_cases, fvs_summary, fvs_carbon, fvs_hrv_carbon, fvs_econharvestvalue, fvs_econsummary]

mydb = "FVSOut"
myusername = 'postgres'
myhost = 'localhost'
conn_string = "dbname={dbname} user={user} host={host}".format(dbname=mydb, user=myusername, host=myhost)

create_tables(conn_string, my_tables)

