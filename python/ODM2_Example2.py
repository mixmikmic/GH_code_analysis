import os
import sqlite3
import sys
import shutil

from yodatools.converter.Inputs.excelInput import ExcelInput
from yodatools.converter.Outputs.dbOutput import dbOutput
from yodatools.converter.Outputs.yamlOutput import yamlOutput

import odm2api.ODM2.models as odm2models

from IPython.core.display import display, HTML

# Check package version used
import sqlalchemy
import yodatools

sqlalchemy.__version__, yodatools.__version__

# Assign directory paths and SQLite file name
dpth = os.getcwd()
dbname_sqlite = "ODM2_Example2.sqlite"

sqlite_pth = os.path.join(dpth, os.path.pardir, "data", dbname_sqlite)

yodaxls_dbname = 'YODA_iUTAH_Specimen_Example.xlsx'

yoda_pth = os.path.join(dpth, os.path.pardir, "data", yodaxls_dbname)
print(yoda_pth)

excel = ExcelInput()

excel.parse(yoda_pth)

session = excel.sendODM2Session()
print("Done parsing Excel file!")

# Get all of the Methods that were loaded from the Excel file
methods = session.query(odm2models.Methods).all()
# Print some of the attributes of the methods
for x in methods:
    print("MethodCode: " + x.MethodCode + ", MethodName: " + x.MethodName + ", MethodTypeCV: " + x.MethodTypeCV)

# Create new ODM2 SQLite database by copying the one created in Example 1

# First check to see if the ODM2 SQLite file already exists from previous runs of this example. 
# If so, delete it.
if os.path.isfile(sqlite_pth):
    os.remove(sqlite_pth)

shutil.copy(os.path.join(dpth, os.path.pardir, "data", "ODM2_Example1.sqlite"), 
            sqlite_pth)

# Write the data to the SQLite database, using the connection string to the ODM2 database defined
dbconn_str = "sqlite:///" + sqlite_pth
do = dbOutput()
do.save(session, dbconn_str)

# Provide a link to the ODM2 SQLite file that the data were written to
print("\nYou can download the ODM2 SQLite database populated with data using the following link:")

# This is hard-wiring a path expectation. 
# Which is fine if we know what file path jupyter was started under

sqlite_relpth = os.path.join(os.path.pardir, "data", dbname_sqlite)
display(HTML('<a href=%s target="_blank">%s<a>' % (sqlite_relpth, dbname_sqlite)))

# Write the output to a YODA file
yodaname = "ODM2_Example2.yaml"

yoda_relpth = os.path.join(os.path.pardir, "data", yodaname)

yo = yamlOutput()
yo.save(session, yoda_relpth)

# Provide a link to download the YODA file created
print("\nYou can download the populated YODA file using the following link:")

display(HTML('<a href=%s target="_blank">%s<a>' % (yoda_relpth, yodaname)))



