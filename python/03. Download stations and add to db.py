import pandas as pd
pd.options.display.max_columns = 999

stations_df = pd.read_excel("http://orr.gov.uk/__data/assets/excel_doc/0019/20179/Estimates-of-Station-Usage-in-2014-15.xlsx",2)

stations_df.head()

headers = [h.lower() for h in list(stations_df.columns)]
headers = [h.replace(" ", "_").replace("(", "").replace(")","") for h in headers]
stations_df.columns = headers
stations_df.head()
stations_df["london_or_gb"] = "gb"
stations_df.loc[stations_df["county_or_unitary_authority"] == "Greater London","london_or_gb"] = "london"
stations_df.head()

# Now write out to postgres
from mylibrary.connections import conn, engine, cursor
stations_df.to_sql("all_stations", engine, schema="tt_gh", if_exists="replace", index=False)

# Create geometry column for the points including a spatial index for efficient querying

sql = """
SELECT AddGeometryColumn ('tt_gh', 'all_stations', 'geom', 27700, 'POINT', 2);
UPDATE tt_gh.all_stations SET geom = ST_GeomFromText('POINT(' || os_grid_easting || ' ' || os_grid_northing || ')', 27700 );
CREATE INDEX idx_geom_all_stations_points ON tt_gh.all_stations USING gist(geom);
"""
cursor.execute(sql)
conn.commit()

#Make a lat and lng column

sql = """

ALTER TABLE tt_gh.all_stations ADD lat float, ADD lng float, ADD icscode text, 
ADD icscode_status text, ADD tfl_request text, ADD tfl_response json, ADD tfl_message  text;
UPDATE tt_gh.all_stations SET
    lng = ST_X(ST_TRANSFORM(geom, 4326)),
    lat = ST_Y(ST_TRANSFORM(geom,4326));
ALTER TABLE tt_gh.all_stations  ADD PRIMARY KEY (nlc);
""" 
cursor.execute(sql)
conn.commit()

sql = """
select * from tt_gh.all_stations limit 5
"""
pd.read_sql(sql, conn)

sql = """
select count(*), london_or_gb  from tt_gh.all_stations group by london_or_gb
"""
pd.read_sql(sql, conn)

