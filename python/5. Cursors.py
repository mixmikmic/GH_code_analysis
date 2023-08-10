# Demonstrate retrieving tuples from a table
import os
from arcpy.da import SearchCursor

# Set the path to your data
cat_table = os.path.join(os.getcwd(), "demo.gdb\\Redlands_Cat_Sightings")

# Run the cursor to print the data
table_data = SearchCursor(cat_table, "*")
for row in table_data:
    print(row)

import os
from arcpy.da import SearchCursor

cat_table = os.path.join(os.getcwd(), "demo.gdb\\Redlands_Cat_Sightings")
fields = "*"
sql_query = "TYPE = 'Orange'"
with SearchCursor(cat_table, fields, sql_query) as table_data:
    for row in table_data:
        print(row)

from arcpy.da import InsertCursor

# Set the path to your data
cat_table = os.path.join(os.getcwd(), "demo.gdb\\Redlands_Cat_Sightings")

rows = [
    (6, (-13045960.543499999, 4036405.2462000027), 'Hairless', 1, 0),
    (7, (-13045952.4474, 4036413.342299998), 'Raccoon', 4, 0)
]

# Run the cursor to insert new rows
fields = "*"
with InsertCursor(cat_table, fields) as table_data:
    for row in rows:
        table_data.insertRow(row)  # Run cell 1 to see the new rows

from arcpy.da import UpdateCursor

fields = "*"
sql_query = "TYPE = 'Virtual'"
with UpdateCursor(cat_table, fields, sql_query) as table_data:
    for row in table_data:
        table_data.deleteRow()  # takes no argument(s), ensure your sql query is correct



