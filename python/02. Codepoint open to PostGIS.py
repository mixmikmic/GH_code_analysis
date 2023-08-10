import pandas as pd

codepoint_dir = r"/Users/robinlinacre/Downloads/codepo_gb"

# Get column names
column_headers_df = pd.read_csv("/Users/robinlinacre/Downloads/codepo_gb/Doc/Code-Point_Open_Column_Headers.csv")
headers = column_headers_df.loc[0]
headers = [h.lower() for h in list(headers)]
headers

# Iterate through the CSVs in codepoint open concatenating them together into one big table
import os 
files = os.listdir(os.path.join(codepoint_dir,"Data/CSV"))

dfs = []
for f in files:
    this_file = os.path.join(codepoint_dir,"Data/CSV", f)
    if ".csv" in this_file:
        this_df = pd.read_csv(this_file, header=None)
        dfs.append(this_df)

final_df = pd.concat(dfs)
final_df.columns = headers
final_df.head()

from mylibrary.connections import engine, cursor, conn
final_df.to_sql("all_postcodes", engine, schema="tt_gh", if_exists="replace")

# Create geometry column for the points including a spatial index for efficient querying

sql = """
SELECT AddGeometryColumn ('tt_gh', 'all_postcodes', 'geom', 27700, 'POINT', 2);
UPDATE tt_gh.all_postcodes SET geom = ST_GeomFromText('POINT(' || eastings || ' ' || northings || ')', 27700 );
CREATE INDEX idx_geom_all_postcodes_tt_gh ON tt.all_postcodes USING gist(geom);
"""

cursor.execute(sql)
conn.commit()

