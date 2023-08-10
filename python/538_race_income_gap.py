# basic setup. You'd do this every time you set out to use pandas with Census Reporter's SQL
import pandas as pd
from sqlalchemy import create_engine 
# for below to work, you must set the PGPASSWORD env variable or have no-password login enabled
engine = create_engine('postgresql://census@localhost:5432/census')

# load in white and black median income for Census places (sumlevel = 160)
white = pd.read_sql_query("select g.geoid, g.name, d.b19013h001 as white                            from acs2013_3yr.geoheader g,                                 acs2013_3yr.b19013h d                                 where d.geoid = g.geoid                                 and g.sumlevel = 160",engine, index_col='geoid')
black = pd.read_sql_query("select g.geoid, d.b19013b001 as black                            from acs2013_3yr.geoheader g,                                 acs2013_3yr.b19013b d                                 where d.geoid = g.geoid                                 and g.sumlevel = 160",engine, index_col='geoid')

# put the parts together and compute the gap
df = white.join(black)
df = df.dropna()
df['gap'] = df.white - df.black
df.sort('gap',ascending=True,inplace=True)

df.rename(columns={'white': 'white_income', 'black': 'black_income'}, inplace=True)
population = pd.read_sql_query("select geoid, b03002001 as total_pop, b03002003 as white_pop,                                 b03002004+b03002014 as black_pop from acs2013_3yr.b03002                                 where geoid like '16000US%%'",
                               engine, index_col='geoid')
df = df.join(population)

df.dropna(inplace=True)
df['black_pop_pct'] = df.black_pop / df.total_pop
# I'm running out of creative names for my variables
df2 = df[(df.black_pop_pct >=.1) & (df.gap < 0)]

df2.head(10)

df2.sort('total_pop',ascending=False).head(3)

df3 = df2[df2.gap*-1 > df2.white_income/2]
df3.sort('total_pop',ascending=False).head() 

# Always clean up your database
engine.dispose()

