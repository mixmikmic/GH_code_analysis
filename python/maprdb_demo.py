# Sample code for using JSON tables in MapR-DB with Python
import json
import sys
import maprdb

# Connect to MapR cluster.
conn = maprdb.connect()

# Declare the path in the MapR filesystem in which to find the table:
TABLE_PATH = "/mapr/my.cluster.com/tmp/crm_data"
# Get a reference to the table:
table = conn.get(TABLE_PATH)

# Read top 2 rows in the table:
import itertools
for item in itertools.islice(table.find(), 2):
        print(item)

# Lookup a specific row in the table by id:
table.find_by_id("43ba751a-72bd-4974-8c5a-32e4356b3776")



