import os
import sys
sys.path.append('../')

import pandas as pd
import sqlalchemy as sqla
from sqlalchemy import create_engine

SBA_DWH = os.getenv('SBA_DWH')
engine = create_engine(SBA_DWH)

from googleplaces import GooglePlaces, types, lang
from utilities import geocoder


YOUR_API_KEY = os.getenv('GOOGLE_PLACES_API')

# This will be abstracted/replaced later (possibly reading from URI)
with engine.begin() as conn:
    df = pd.read_sql_table('sba_sfdo', conn, 'stg_analytics')
df['address'] = df.borr_street + ',' + df.borr_city + ',' + df.borr_state + ',' + df.borr_zip.map(str)

df = geocoder.geocode(df=df, api_key=YOUR_API_KEY)

df.head()



