# general use imports
import datetime
import glob
import inspect
import numpy
import os
import six
import warnings

# pandas-related imports
import pandas
import sqlalchemy

# CSV file reading-related imports
import csv

# database interaction imports
import psycopg2
import psycopg2.extras

print( "Imports loaded at " + str( datetime.datetime.now() ) )

# schema name
schema_name = "ildoc"

# admin role
admin_role = schema_name + "_admin"
select_role = schema_name + "_select"

# ==> database table names - just like file names above, store reused database information in variables here.

# work table name
work_db_table = "person"

print( "Database variables initialized at " + str( datetime.datetime.now() ) )

# Database connection properties
db_host = "10.10.2.10"
db_port = -1
db_username = None
db_password = None
db_name = "appliedda"

print( "Database connection properties initialized at " + str( datetime.datetime.now() ) )

# initialize database connections
# Create connection to database using SQLAlchemy
#     (3 '/' indicates use enviroment settings for username, host, and port)
sqlalchemy_connection_string = "postgresql://"

if ( ( db_host is not None ) and ( db_host != "" ) ):
    sqlalchemy_connection_string += str( db_host )
#-- END check to see if host --#

sqlalchemy_connection_string += "/"

if ( ( db_name is not None ) and ( db_name != "" ) ):
    sqlalchemy_connection_string += str( db_name )
#-- END check to see if host --#

# create engine.
pgsql_engine = sqlalchemy.create_engine( sqlalchemy_connection_string )

print( "SQLAlchemy engine created at " + str( datetime.datetime.now() ) )

# create psycopg2 connection to Postgresql

# example connect() call that uses all the possible parameters
#pgsql_connection = psycopg2.connect( host = db_host, port = db_port, database = db_name, user = db_username, password = db_password )

# for SQLAlchemy, just needed database name. Same for DBAPI?
pgsql_connection = psycopg2.connect( host = db_host, database = db_name )

print( "Postgresql connection to database \"" + db_name + "\" created at " + str( datetime.datetime.now() ) )

# create a cursor that maps column names to values
pgsql_cursor = pgsql_connection.cursor( cursor_factory = psycopg2.extras.DictCursor )

print( "Postgresql cursor for database \"" + db_name + "\" created at " + str( datetime.datetime.now() ) )

# rollback, in case you need it.
pgsql_connection.rollback()

print( "Postgresql connection for database \"" + db_name + "\" rolled back at " + str( datetime.datetime.now() ) )

# Create table - declare variables
table_name = ""
table_name = work_db_table

# generate SQL
sql_string = "CREATE TABLE " + schema_name + "." + table_name

# add columns
sql_string += " ("
sql_string += " id BIGSERIAL PRIMARY KEY"
sql_string += ", recptno bigint"
sql_string += ", sex bigint"
sql_string += ", rac bigint"
sql_string += ", rootrace bigint"
sql_string += ", foreignbn bigint"
sql_string += ", ssn_hash text"
sql_string += ", fname_hash text"
sql_string += ", lname_hash text"
sql_string += ", birth_date date"
sql_string += " )"
sql_string += ";"

print( "====> " + str( sql_string ) )

# run SQL
pgsql_cursor.execute( sql_string )
pgsql_connection.commit()

#temp_df = pandas.read_sql( sql_string, con = pgsql_engine )
#temp_length = len( temp_df )
#temp_df.head( n = temp_length )

print( "====> " + str( sql_string ) + " completed at " + str( datetime.datetime.now() ) )

# Create table - declare variables
table_name = ""
table_name = work_db_table

# generate SQL
sql_string = "CREATE TABLE " + schema_name + "." + table_name
sql_string += " AS SELECT"
sql_string += " recptno"
sql_string += ", sex"
sql_string += ", rac"
sql_string += ", rootrace"
sql_string += ", foreignbn"
sql_string += ", ssn_hash"
sql_string += ", fname_hash"
sql_string += ", lname_hash"
sql_string += ", birth_date"
sql_string += " FROM idhs.hh_member"
#sql_string += " WHERE EXTRACT( year FROM birth_date ) = 1976"
#sql_string += " LIMIT 1000"
sql_string += ";"

print( "====> " + str( sql_string ) )

# run SQL
pgsql_cursor.execute( sql_string )
pgsql_connection.commit()

#temp_df = pandas.read_sql( sql_string, con = pgsql_engine )
#temp_length = len( temp_df )
#temp_df.head( n = temp_length )

print( "====> " + str( sql_string ) + " completed at " + str( datetime.datetime.now() ) )

# UPDATE ownership - declare variables
table_name = ""
table_name = work_db_table

# generate SQL
sql_string = "ALTER TABLE " + schema_name + "." + table_name
sql_string += " OWNER TO " + admin_role
sql_string += ";"

# run SQL
pgsql_cursor.execute( sql_string )
pgsql_connection.commit()

#temp_df = pandas.read_sql( sql_string, con = pgsql_engine )
#temp_length = len( temp_df )
#temp_df.head( n = temp_length )

print( "====> " + str( sql_string ) + " completed at " + str( datetime.datetime.now() ) )

# admin_role privileges - declare variables
table_name = ""
table_name = work_db_table

# generate SQL
sql_string = "GRANT ALL PRIVILEGES ON TABLE " + schema_name + "." + table_name
sql_string += " TO " + admin_role
sql_string += ";"

# run SQL
pgsql_cursor.execute( sql_string )
pgsql_connection.commit()

#temp_df = pandas.read_sql( sql_string, con = pgsql_engine )
#temp_length = len( temp_df )
#temp_df.head( n = temp_length )

print( "====> " + str( sql_string ) + " completed at " + str( datetime.datetime.now() ) )

# select_role privileges - declare variables
table_name = ""
table_name = work_db_table

# generate SQL
sql_string = "GRANT SELECT ON TABLE " + schema_name + "." + table_name
sql_string += " TO " + select_role
sql_string += ";"

# run SQL
pgsql_cursor.execute( sql_string )
pgsql_connection.commit()

#temp_df = pandas.read_sql( sql_string, con = pgsql_engine )
#temp_length = len( temp_df )
#temp_df.head( n = temp_length )

print( "====> " + str( sql_string ) + " completed at " + str( datetime.datetime.now() ) )

# Create table - declare variables
existing_table_name = ""
new_table_name = existing_table_name + "_001"

# generate SQL
sql_string = "CREATE TABLE " + schema_name + "." + new_table_name
sql_string += " AS SELECT"
sql_string += " existing.recptno"
sql_string += ", existing.sex"
sql_string += ", existing.rac"
sql_string += ", existing.rootrace"
sql_string += ", existing.foreignbn"
sql_string += ", existing.ssn_hash"
sql_string += ", existing.fname_hash"
sql_string += ", existing.lname_hash"
sql_string += ", existing.birth_date"
sql_string += ", ildoc.ildoc_docnbr"
sql_string += " FROM " + schema_name + "." + existing_table_name + " existing"
sql_string += " LEFT OUTER JOIN ildoc.person ildoc ON ( existing.ssn_hash = ildoc.ssn_hash )"
#sql_string += " WHERE EXTRACT( year FROM birth_date ) = 1976"
#sql_string += " LIMIT 1000"
sql_string += ";"

print( "====> " + str( sql_string ) )

# run SQL
pgsql_cursor.execute( sql_string )
pgsql_connection.commit()

#temp_df = pandas.read_sql( sql_string, con = pgsql_engine )
#temp_length = len( temp_df )
#temp_df.head( n = temp_length )

print( "====> " + str( sql_string ) + " completed at " + str( datetime.datetime.now() ) )

# Create table - declare variables
existing_table_name = ""
new_table_name = existing_table_name + "_001"

# generate SQL
sql_string = "DROP TABLE " + schema_name + "." + existing_table_name
sql_string += ";"

print( "====> " + str( sql_string ) )

# run SQL
pgsql_cursor.execute( sql_string )
pgsql_connection.commit()

#temp_df = pandas.read_sql( sql_string, con = pgsql_engine )
#temp_length = len( temp_df )
#temp_df.head( n = temp_length )

print( "====> " + str( sql_string ) + " completed at " + str( datetime.datetime.now() ) )

# Add column
table_name = ""
column_name = "<column_name>"
column_type = "<column_type>"

# generate SQL
sql_string = "ALTER TABLE " + schema_name + "." + table_name

# start date values
sql_string += " ADD COLUMN " + column_name + " " + column_type

sql_string += ";"

# run SQL
pgsql_cursor.execute( sql_string )
pgsql_connection.commit()

#temp_df = pandas.read_sql( sql_string, con = pgsql_engine )
#temp_length = len( temp_df )
#temp_df.head( n = temp_length )

print( "====> " + str( sql_string ) + " completed at " + str( datetime.datetime.now() ) )

# ==> Set `start_date` from `start_date_orig`.

# declare variables
current_column = "start_date"

# UPDATE
sql_string = "UPDATE " + schema_name + "." + work_db_table
sql_string += " SET " + current_column + " = TO_DATE( " + current_column + "_orig, 'YYYY-MM-DD' )"

# WHERE clause
# WHERE clause
where_clause = "WHERE " + current_column + " IS NULL"
where_clause += " AND ( ( " + current_column + "_orig IS NOT NULL ) AND ( " + current_column + "_orig != '' ) )"
sql_string += " " + where_clause

sql_string += ";"

print( "SQL: " + sql_string )

# run SQL
pgsql_cursor.execute( sql_string )
pgsql_connection.commit()

print( "UPDATEd " + where_clause + " at " + str( datetime.datetime.now() ) )

# declare variables
sql_string = ""
id_column_name = ""
recptno_value = None
ssn_hash_value = None
bdate_year_value = None
work_cursor = None
work_year_18 = None
work_year_19 = None
work_year_20 = None
work_year_21 = None
work_year_in_list = []
in_q3_year_list = []
non_q3_year_list = []
years_worked_in_q3 = -1
years_worked_non_q3 = -1
years_worked_only_q3 = -1
row_counter = -1

# declare variables working with work cursor.
work_sql_string = ""
work_row = None
wage_year = None
wage_quarter = None

# make a work cursor, so you can query and update independent of your loop over your people.
work_cursor = pgsql_connection.cursor( cursor_factory = psycopg2.extras.DictCursor )

# get IDs from work table.
sql_string = "SELECT * FROM " + schema_name + "." + work_db_table

# only get rows that have not yet been updated - this lets you pick up if the program is interrupted.
#sql_string += " WHERE years_worked_in_q3 IS NULL"

sql_string += ";"

print( sql_string )

# get list of records in person file, so we can process one-by-one
pgsql_cursor.execute( sql_string )
row_counter = 0
for current_row in pgsql_cursor:
    
    # increment row Counter
    row_counter += 1
    
    # initialize variables to make sure we empty out from last row.
    recptno_value = None
    ssn_hash_value = None
    bdate_year_value = None
    work_year_18 = -1
    work_year_19 = -1
    work_year_20 = -1
    work_year_21 = -1
    work_year_in_list = []
    in_q3_year_list = []
    non_q3_year_list = []
    work_sql_string = ""
    years_worked_in_q3 = -1
    years_worked_non_q3 = -1
    years_worked_only_q3 = -1
    
    # get values from record
    recptno_value = current_row.get( "recptno", None )
    ssn_hash_value = current_row.get( "ssn_hash", None )
    bdate_year_value = current_row.get( "bdate_year", None )
    
    # for that recipient, perform logic to derive value for recipient.
    
    # Example: number of years worked Q3 for work_years 18-21,
    #     number of years worked in quarters outside of Q3,
    #     and number of years worked only in Q3.
    work_year_18 = bdate_year_value + 18
    work_year_19 = work_year_18 + 1
    work_year_20 = work_year_19 + 1
    work_year_21 = work_year_20 + 1
    
    # make list of work years, converted to strings for use in query.
    work_year_in_list = [ str( work_year_18 ), str( work_year_19 ), str( work_year_20 ), str( work_year_21 ) ]
    
    # get all wage records for this person in these years.
    in_q3_year_list = []
    non_q3_year_list = []
    
    # create SQL to retrieve wage records...
    work_sql_string = "SELECT * FROM ides.il_wage"
    
    # ...for the current person...
    work_sql_string += " WHERE ssn = '" + ssn_hash_value + "'"
    
    # ...in the specified years...
    work_sql_string += " AND year IN ( " + ", ".join( work_year_in_list ) + " )"
    
    # ...ordered by year and quarter, ascending.
    work_sql_string += " ORDER BY year ASC, quarter ASC"

    work_sql_string += ";"
    
    # call the query and loop over results
    work_cursor.execute()
    for work_row in work_cursor:
        
        # ==> Here you do whatever work you need to for a given wage record.
        
        # get data
        wage_year = work_row.get( "year", None )
        wage_quarter = work_row.get( "quarter", None )
        
        # quarter 3?
        if wage_quarter == 3:
            
            # 3 - if year not already in list, add it (can have multiple rows per quarter, so don't want to count)
            if wage_year not in in_q3_year_list:
                
                in_q3_year_list.append( wage_year )
                
            #-- END check to see if we need to add year. --#
            
        else:
            
            # 1, 2, or 4 - if year not already in list, add it (can have multiple rows per quarter, so don't want to count)
            if wage_year not in non_q3_year_list:
                
                non_q3_year_list.append( wage_year )
                
            #-- END check to see if we need to add year. --#
            
        #-- END check what quarter. --#
        
    #-- END loop over wage records. --#
    
    # ==> calculate values you care about:
    
    years_worked_in_q3 = len( in_q3_year_list )
    years_worked_non_q3 = len( non_q3_year_list )
    years_worked_only_q3 = 0
    
    # loop over q3 year list
    for current_year in in_q3_year_list:
        
        # is that year also in non_q3 list?
        if current_year not in non_q3_year_list:
            
            # no - just q3
            years_worked_only_q3 += 1
            
        #-- END check to see if q3 year in non-q3 year list. --#
        
    #-- END loop over years in Q3 list. --#
    
    # ==> UPDATE
    work_sql_string = "UPDATE " + schema_name + "." + work_db_table
    work_sql_string += " SET years_worked_in_q3 = " + str( years_worked_in_q3 )
    work_sql_string += ", years_worked_non_q3 = " + str( years_worked_non_q3 )
    work_sql_string += ", years_worked_only_q3 = " + str( years_worked_only_q3 )
    work_sql_string += " WHERE recptno = " + str( recptno_value )
    work_sql_string += ";"
    
    # execute and commit.
    work_cursor.execute()
    pgsql_connection.commit()

    # every <cluster_size> people, output a message
    cluster_size = 1000
    if ( ( row_counter % cluster_size ) == 0 ):
        
        print( "Processsed " + str( row_counter ) + " people." )
        
        # if you want, you could also only commit every <cluster_size> UPDATES.
        # - in some cases, this will improve performance.
        # To do this, commend out the commit above, then uncomment this one.
        #pgsql_connection.commit()
        
    #-- END check to see if this is a multiple of <cluster_size> --#
    
#-- END loop over people. --#

