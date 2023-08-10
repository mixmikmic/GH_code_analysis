# imports
import datetime
import psycopg2
import psycopg2.extras

print( "psycopg2 imports completed at " + str( datetime.datetime.now() ) )

# Connect...
pgsql_connection = None

# set up connection properties
db_host = "10.10.2.10"
db_database = "appliedda"

# and connect.
pgsql_connection = psycopg2.connect( host = db_host, database = db_database )

print( "psycopg2 connection to host: " + db_host + ", database: " + db_database + " completed at " + str( datetime.datetime.now() ) )

# ...and create cursor.
pgsql_cursor = None

# results come back as a list of columns:
#pgsql_cursor = pgsql_connection.cursor()

# results come back as a dictionary where values are mapped to column names (preferred)
pgsql_cursor = pgsql_connection.cursor( cursor_factory = psycopg2.extras.DictCursor )

print( "psycopg2 cursor created at " + str( datetime.datetime.now() ) )

# Single row query
sql_string = ""
result_row = None

# SQL
sql_string = "SELECT COUNT( * ) AS row_count FROM idhs.hh_member;"

# execute it.
pgsql_cursor.execute( sql_string )

# fetch first (and only) row, then output the count
first_row = pgsql_cursor.fetchone()
print( "row_count = " + str( first_row[ "row_count" ] ) )

# Multiple row query
sql_string = ""
result_list = None
result_row = None
row_counter = -1

# SQL
sql_string = "SELECT * FROM idhs.hh_member LIMIT 1000;"

# execute it.
pgsql_cursor.execute( sql_string )

# ==> fetch rows to loop over:

# all rows.
#result_list = pgsql_cursor.fetchall()

# first 10 rows.
result_list = pgsql_cursor.fetchmany( size = 10 )

# loop
result_counter = 0
for result_row in result_list:
    
    result_counter += 1
    print( "- row " + str( result_counter ) + ": " + str( result_row ) )
    
#-- END loop over 10 rows --#

# ==> loop over the rest one at a time.
result_counter = 0
result_row = pgsql_cursor.fetchone()
while result_row is not None:
    
    # increment counter
    result_counter += 1
    
    # get next row
    result_row = pgsql_cursor.fetchone()
    
#-- END loop over rows, one at a time. --#

print( "fetchone() row_count = " + str( result_counter ) )

# Close Connection and cursor
pgsql_cursor.close()
pgsql_connection.close()

print( "psycopg2 cursor and connection closed at " + str( datetime.datetime.now() ) )

# imports
import sqlalchemy
import datetime

# Connect
connection_string = 'postgresql://10.10.2.10/appliedda'
pgsql_engine = sqlalchemy.create_engine( connection_string )

print( "SQLAlchemy engine connected to " + connection_string + " at " + str( datetime.datetime.now() ) )

# Single row query - with the streaming option so it does not return results until we "fetch" them:
sql_string = "SELECT COUNT( * ) AS row_count FROM idhs.hh_member;"
query_result = pgsql_engine.execution_options( stream_results = True ).execute( sql_string )

# output results - you can also check what columns "query_result" has by accessing
#     it's "keys" since it is just a Python dict object. Like so:
print( query_result.keys() )

# print an empty string to separate out our two more useful print statements
print('')

# fetch first (and only) row, then output the count
first_row = query_result.fetchone()
print("row_count = " + str( first_row[ "row_count" ] ) )

# Multiple row query
sql_string = ""
query_result = None
result_list = None
result_row = None
row_counter = -1

# run query with the streaming option so it does not return results until we "fetch" them:

# SQL
sql_string = "SELECT * FROM idhs.hh_member LIMIT 1000;"

# execute it.
query_result = pgsql_engine.execution_options( stream_results = True ).execute( sql_string )

# ==> fetch rows to loop over:

# all rows.
#result_list = query_result.fetchall()

# first 10 rows.
result_list = query_result.fetchmany( size = 10 )

# loop
result_counter = 0
for result_row in result_list:
    
    result_counter += 1
    print( "- row " + str( result_counter ) + ": " + str( result_row ) )
    
#-- END loop over 10 rows --#

# ==> loop over the rest one at a time.
result_counter = 0
result_row = query_result.fetchone()
while result_row is not None:
    
    # increment counter
    result_counter += 1
    
    # get next row
    result_row = query_result.fetchone()
    
#-- END loop over rows, one at a time. --#

print( "fetchone() row_count = " + str( result_counter ) )

# Clean up:
pgsql_engine.dispose()

print( "SQLAlchemy engine dispose() called at " + str( datetime.datetime.now() ) )

# imports
import datetime
import pandas

# Connect - create SQLAlchemy engine for pandas to use.
connection_string = 'postgresql://10.10.2.10/appliedda'
pgsql_engine = sqlalchemy.create_engine( connection_string )

print( "SQLAlchemy engine connected to " + connection_string + " at " + str( datetime.datetime.now() ) )

# Single row query
sql_string = ""
df_ildoc_admit = ""
first_row = None
row_count = -1

# Single row query
sql_string = "SELECT COUNT( * ) AS row_count FROM idhs.hh_member;"
df_ildoc_admit = pandas.read_sql( sql_string, con = pgsql_engine )

# get row_count - first get first row
first_row = df_ildoc_admit.iloc[ 0 ]

# then grab value.
row_count = first_row[ "row_count" ]

print("row_count = " + str( row_count ) )

# and call head().
df_ildoc_admit.head()

# Multiple row query
sql_string = ""
df_ildoc_admit = ""
row_count = -1
result_row = None

# SQL
sql_string = "SELECT * FROM idhs.hh_member LIMIT 1000;"

# execute it.
df_ildoc_admit = pandas.read_sql( sql_string, con = pgsql_engine )

# unlike previous Python examples, rows are already fetched and in a dataframe:

# you can loop over them...
row_count = 0
for result_row in df_ildoc_admit.iterrows():
    
    row_count += 1
    
#-- END loop over rows. --#

print( "loop row_count = " + str( row_count ) )

# Print out the first X using head()
output_count = 10
df_ildoc_admit.head( output_count )

# etc.

# Close Connection - Except you don't have to becaue pandas does it for you!

