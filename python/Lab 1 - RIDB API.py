import pandas as pd 
import numpy as np
import config
import requests
import json
from pandas.io.json import json_normalize

ridb_facilities_url = "https://ridb.recreation.gov/api/v1/facilities"

camping_params = params=dict(activity_id=9, apiKey = config.RIDB_API_KEY,                             latitude=45.4977712, longitude=-121.8211673, radius=15)
response = requests.get(ridb_facilities_url,camping_params)
camping_json  = json.loads(response.text)
camping_df = json_normalize(camping_json['RECDATA'])

camping_df.FacilityName

mock_url = "http://" + config.LAMP_IP + "/ridb_mock.json"
camping_df = pd.read_json(mock_url)

camping_df.head()

camping_df.LastUpdatedDate.unique()

camping_df.LegacyFacilityID.unique()

camping_df.shape

camping_df = camping_df.replace('', np.nan)

#camping_df = camping_df.drop(['GEOJSON.COORDINATES','GEOJSON.TYPE'], axis=1)

camping_df[camping_df.FacilityLatitude.isnull()]

camping_df = camping_df.dropna(subset=['FacilityLatitude','FacilityLongitude'])

camping_df[camping_df.FacilityLatitude.isnull()]

camping_df.shape

camping_df.to_csv('test.csv', index=False)

csv_test = pd.read_csv('test.csv')

csv_test.head()

from sqlalchemy import create_engine
connectStr = "mysql+pymysql://" + config.DB_USER + ":" + config.DB_PASS + "@" + config.DB_HOST +  "/" + config.DB_NAME
engine =create_engine(connectStr)

camping_df.to_sql('test',engine,if_exists='replace')

sql_test = pd.read_sql('select * from test', engine, index_col='index')

sql_test.head()

sql_test.LastUpdatedDate.unique()

sum(sql_test.LegacyFacilityID.isnull())



