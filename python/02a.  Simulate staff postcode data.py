import pandas as pd 
from mylibrary.connections import cursor, conn, engine, Automapped_Base, session
sql = """
drop table if exists tt_gh.staff_locations;
create table  tt_gh.staff_locations as
(select postcode, s.geom from tt_gh.all_postcodes as s, tt_gh.gb_and_london as g
where st_contains(g.geom,s.geom)  and g.name='london'
order by random()
limit 1000)

union all

(select postcode, s.geom from tt_gh.all_postcodes as s, tt_gh.gb_and_london as g
where not st_contains(g.geom,s.geom)  and g.name='london'
order by random()
limit 1000);

CREATE INDEX idx_geom_staff_locations ON tt_gh.staff_locations USING gist(geom);
"""
cur = conn.cursor()
cur.execute(sql)
conn.commit()

