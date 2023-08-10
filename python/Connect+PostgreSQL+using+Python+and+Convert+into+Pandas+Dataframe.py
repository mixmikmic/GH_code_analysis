import psycopg2 #library for connecting Postgresql
import pandas.io.sql as psql #library to use sql command and convert extraction into dataframe
import pandas as pd # Pandas Dataframe
from datetime import datetime #Convert timestamp in datetime formart

# Create a Dynamic function - With Inputs 
# Here Evaluation_Id is the input parameter as it is specific to each client

def database_extract(evaluation_id):
    format_list = [evaluation_id, ] # This is used in SQL extraction, As SQL command cannot take dynamic inputs, we need to build the command first and then execute
    
    # Database Parameters
    database = "xxxxxxxxxxxxx"
    hostname = "xxxxxxxxxxx.compute-1.amazonaws.com"
    port = 5432 
    username = "xxxxxxxxxxxx" 
    password = "xxxxxxxxxxxxxxxx"
    
    #Establish connection with the database
    myConnection = psycopg2.connect( host=hostname, user=username, password=password, dbname=database )  
    
    # Read the SQL query and save in the dataframe format
    #.format at the end of command is used to build the query first
    # "{}" in the query would be replaced by EvaluationID before execution of query
    dataframe = psql.read_sql("SELECT XXXXXXXXXXX".format(*format_list), myConnection)
    
    # rename the column names 
   
    dataframe.columns = ['evaluation_id', 'xxxxxxx', 'xxxxxxxx']
    
    return(dataframe)

# Save the output of the function in the variable 
df_postgresql = database_extract(evaluation_id = 99999)

