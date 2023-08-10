get_ipython().run_cell_magic('bash', '', 'shp2pgsql -I -s 27700 /Users/robinlinacre/Documents/python_projects/moj_national/shapefiles/gb_london_simplified_final.shp tt.gb_and_london | psql -d postgres')

# The shapefile doesn't contain the right attributes - this adds them.
from mylibrary.connections import  cursor, conn
sql = """
delete from tt.gb_and_london where gid = 3;
alter table tt.gb_and_london ADD name text;
update  tt.gb_and_london
set name = 'london' where gid = 1;
update  tt.gb_and_london
set name = 'gb' where gid = 2;
"""
cursor.execute(sql)
conn.commit()

