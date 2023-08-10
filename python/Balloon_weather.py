import pandas as pd
citypath ="C:\\Users\\wesle\\Desktop\\contest\\data\\CityData.csv"
Insitupath="C:\\Users\\wesle\\Desktop\\contest\\data\\In_situMeasurementforTraining_201712.csv"
Trainpath="C:\\Users\\wesle\\Desktop\\contest\\data\\ForecastDataforTraining_201712.csv"

city=pd.read_csv(citypath)
# city location 0 represent london is the starting point for all balloons
city

Insitu=pd.read_csv(Insitupath)
Insitu.head() # Real-time wind value 

Insitu.describe()

Train=pd.read_csv(Trainpath)
Train.head()

Train.describe()

Train1=Train[Train.date_id==1]# subset only one day

newtrain1=Train1.pivot_table(values="wind",index=["xid","yid","date_id","hour"],columns="model")
newtrain1.head()

newdf=df.pivot_table(values='value', index=['year', 'month'], columns='item')

newdf=df.pivot_table(values='value', index=['year', 'month'], columns='item')
newdf

print(type(newdf))
print(type(df))

