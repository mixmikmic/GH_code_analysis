import os
import sqlite3

from IPython.core.display import display, HTML

# Assign directory paths and SQLite file name
dpth = os.getcwd()
dbname_sqlite = "ODM2_Example1.sqlite"

sqlite_pth = os.path.join(dpth, os.path.pardir, "data", dbname_sqlite)

# First check to see if the ODM2 SQLite file already exists from previous runs of this example. If so, delete it.
if os.path.isfile(sqlite_pth):
    os.remove(sqlite_pth)

# Create a new SQLite database and get a cursor
conn = sqlite3.connect(sqlite_pth)
c = conn.cursor()

# Open the DDL SQL file for ODM2
ODM2SQLiteLoad_pth = os.path.join(dpth, os.path.pardir, "code", 'ODM2_for_SQLite.sql')
with open(ODM2SQLiteLoad_pth, 'r') as sqlf:
    sqlString = sqlf.read()

# Execute the DDL SQL script on the blank SQLite database
c.executescript(sqlString)

# Close the connection to the database
conn.close()

print("Done creating ODM2 database!")

# Run the CV Loader script

# note the need to have 3 slashes!
dbconn_str = "sqlite:///" + sqlite_pth

get_ipython().run_line_magic('run', '../code/cvload.py $dbconn_str')
    
print("Done loading controlled vocabularies!")

conn = sqlite3.connect(sqlite_pth)
c = conn.cursor()
cvName = 'CV_SiteType'
sqlString = 'SELECT Name, Definition FROM ' + cvName
c.execute(sqlString)
rows = c.fetchall()

print(rows[0])

conn.close()

print("\nYou can download the blank ODM2 SQLite database with populated CVs using the following link:")

# This is hard-wiring a path expectation. 
# Which is fine if we know what file path jupyter was started under

sqlite_relpth = os.path.join(os.path.pardir, "data", dbname_sqlite)
display(HTML('<a href=%s target="_blank">%s<a>' % (sqlite_relpth, dbname_sqlite)))



