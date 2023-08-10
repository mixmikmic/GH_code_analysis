import numpy as np
import pandas as pd
from pandas import Series,DataFrame

dframe = pd.read_csv('data/innermelbourne.csv')
dframe.columns = dframe.columns.str.strip()

#Dropping columns deemed irrelevant
aframe = dframe.drop(['gcc_name11','gcc_code11','sa2_5dig11','sa1_7dig11','sa3_code11','sa4_code11','ste_code11','ste_name11'],axis=1)

#Group by SA2 suburb
avg_sa2 = aframe[['sa2_name11','SumZScore']].groupby('sa2_name11').mean()

avg_sa2 = avg_sa2.reset_index()

avg_sa2.columns = ['area_name','Walkability Index']

avg_sa2

sa2_health_risk = pd.read_csv('data/sa2_health_risk.csv')
sa2_health_risk.columns = sa2_health_risk.columns.str.strip()

sa2_features = pd.merge(avg_sa2,sa2_health_risk,how='inner',left_on='area_name',right_on='area_name',sort=False)

sa2_features

sa2_life_satisfaction = pd.read_csv('data/sa2_avg_life_satisfaction.csv')
sa2_life_satisfaction.columns = sa2_life_satisfaction.columns.str.strip()

sa2_features = pd.merge(sa2_features,sa2_life_satisfaction,how='inner',left_on='area_name',right_on='area_name',sort=False)

sa2_features.drop(['area_code_x','area_code_y'],axis=1,inplace=True) #Dropping irrelevant columns

sa2_house_transport = pd.read_csv('data/sa2_house_transport.csv')
sa2_house_transport.columns = sa2_house_transport.columns.str.strip()
sa2_house_transport.drop('area_code',axis=1,inplace=True)

sa2_features = pd.merge(sa2_features,sa2_house_transport,how='inner',left_on='area_name',right_on='area_name',sort=False)

sa2_income_var = pd.read_csv('data/sa2_income_var.csv')
sa2_income_var.columns = sa2_income_var.columns.str.strip()
sa2_income_var.drop('area_code',axis=1,inplace=True)

sa2_features = pd.merge(sa2_features,sa2_income_var,how='inner',left_on='area_name',right_on='area_name',sort=False)

sa2_median_avg = pd.read_csv('data/sa2_median_avg.csv')
sa2_median_avg.columns = sa2_median_avg.columns.str.strip()
sa2_median_avg.drop('area_code',axis=1,inplace=True)

sa2_features = pd.merge(sa2_features,sa2_median_avg,how='inner',left_on='area_name',right_on='area_name',sort=False)

#Reading and processing the bus stop csv, aggregating by count.

sa2_bus = pd.read_csv('data/sa2_bus_stops.csv')
sa2_bus.columns = sa2_bus.columns.str.strip()
sa2_bus = sa2_bus[['area_name','METLINKSTOPID']].groupby('area_name').count()
sa2_bus.columns = ['Bus Stop Count']
sa2_bus.reset_index(inplace=True)

#Reading and processing the tram stop csv, aggregating by count.

sa2_tram = pd.read_csv('data/sa2_tram_stops.csv')
sa2_tram.columns = sa2_tram.columns.str.strip()
sa2_tram = sa2_tram[['area_name','METLINKSTOPID']].groupby('area_name').count()
sa2_tram.columns = ['Tram Stop Count']
sa2_tram.reset_index(inplace=True)

#Reading and processing the train stop csv, aggregating by count.

sa2_train = pd.read_csv('data/sa2_train_stops.csv')
sa2_train.columns = sa2_train.columns.str.strip()
sa2_train = sa2_train[['area_name','METLINKSTOPID']].groupby('area_name').count()
sa2_train.columns =['Tran Station Count']
sa2_train.reset_index(inplace=True)

sa2_features = pd.merge(sa2_features,sa2_bus,how='left',left_on='area_name',right_on='area_name',sort=False)
sa2_features = pd.merge(sa2_features,sa2_tram,how='left',left_on='area_name',right_on='area_name',sort=False)
sa2_features = pd.merge(sa2_features,sa2_train,how='left',left_on='area_name',right_on='area_name',sort=False)

sa2_features.fillna(value=0,axis=1,inplace=True)
sa2_features

sa2_features.to_csv('data/sa2_cor_features.csv')

