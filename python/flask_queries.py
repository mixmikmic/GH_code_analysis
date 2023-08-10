import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, inspect
import pandas as pd
import numpy as np
import json

get_ipython().system('ls')

engine = create_engine('sqlite:///refugees_and_conflict.sqlite')

Base = automap_base()

Base.prepare(engine, reflect = True)

Base.classes.keys()

session = Session(engine)

Asylum = Base.classes.asylum
BattleDeaths = Base.classes.battle_deaths
Origin = Base.classes.origin
InfoTable = Base.classes.info_table

samples_df = pd.read_sql_table('asylum', engine)
samples_df.head()

country_names = session.query(Asylum.country_name).all()
country_names = [x[0] for x in country_names]
country_names

samples_df = pd.read_sql_table('info_table', engine)
samples_df.head()

first_row_samples = session.query(InfoTable).first()
first_row_samples.__dict__

info_table = session.query(InfoTable.country_name, InfoTable.gdp_YR2015, InfoTable.population_YR2016, 
                           InfoTable.asylum_YR2016, InfoTable.origin_YR2016).all()

info_table = pd.DataFrame(info_table)
info_table.head()

info_table = info_table.set_index('country_name').to_dict('index')

info_table



