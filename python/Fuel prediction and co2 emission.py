#importing the necessacary packages in python
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import statsmodels.formula.api as smf
get_ipython().magic('matplotlib inline')

elec = pd.read_csv("fuel consumption dataset\MY2012-2017 Battery Electric Vehicles.csv")
elec.head() #display the column names and sample data

cols = [u'MODEL', u'MAKE', u'MODEL.1', u'VEHICLE CLASS', u'FUEL', u'CONSUMPTION', u'Unnamed: 8', u'Unnamed: 9',
       u'Unnamed: 10', u'Unnamed: 11', u'Unnamed: 12', u'CO2 EMISSIONS', u'CO2 ', u'SMOG', u'RANGE', u'RECHARGE']
elec=elec[cols] #select only the columns we need
elec 

newcols = {'MODEL': 'year', 'MAKE':'make', 'MODEL.1':'model', 'VEHICLE CLASS':'vclass', 'FUEL':'fuel', 'CONSUMPTION':'cityElec', 'Unnamed: 8':'hwyElec', 'Unnamed: 9':'combElec',
       'Unnamed: 10':'cityGas', 'Unnamed: 11':'hwyGas', 'Unnamed: 12':'combGas', 'CO2 EMISSIONS':'co2', 'CO2 ':'co2Rate', 'SMOG':'smogRate', 'RANGE':'dist', u'RECHARGE':'recharge'}
elec.rename(columns=newcols, inplace=True)
elec

elec = elec.drop(elec.index[0])
elec

elec.describe()

elec.vclass.unique()

df = pd.read_csv("fuel consumption dataset\Original MY2000-2014 Fuel Consumption Ratings (2-cycle).csv")
# take a look at the dataset
df.head()

df.columns

df.describe()

cols = ['MODEL','VEHICLE CLASS', 'CO2 EMISSIONS']
df=df[cols]
newcols = {
    'MODEL': 'year', 
    'VEHICLE CLASS':'vclass', 
    'CO2 EMISSIONS':'co2'}
df.rename(columns=newcols, inplace=True)
df=df.drop(df.index[0])
df = df.reset_index(drop=True)
df

df['co2'] =  df['co2'].astype(float)
df['vclass']=df['vclass'].astype(str)
df['year'] =  df['year'].astype(int)
df.describe()

df = df.drop( df[ (df.vclass !='SUBCOMPACT' ) & (df.vclass !='MID-SIZE' ) & (df.vclass !='COMPACT' ) & (df.vclass !='TWO-SEATER' ) & (df.vclass !='FULL-SIZE' ) & (df.vclass !='STATION WAGON - SMALL' ) & (df.vclass !='SUV - STANDARD' )].index )
df.index = range(len(df))
df['vclass'] = df['vclass'].replace('SUBCOMPACT', '1')
df['vclass'] = df['vclass'].replace('MID-SIZE', '2')
df['vclass'] = df['vclass'].replace('COMPACT', '3')
df['vclass'] = df['vclass'].replace('TWO-SEATER', '4')
df['vclass'] = df['vclass'].replace('FULL-SIZE', '5')
df['vclass'] = df['vclass'].replace('STATION WAGON - SMALL', '6')
df['vclass'] = df['vclass'].replace('SUV - STANDARD', '7')
df['vclass'] = df['vclass'].astype(int)
df

model = smf.ols(formula='co2 ~ vclass + year', data=df).fit()
model.paramslm = smf.ols(formula='co2 ~ vclass + year', data=df).fit()
model.params #parameters

model.summary()

toPredict = pd.DataFrame({'vclass': [6], 'year':[2000]})
toPredict.head()

prediction=model.predict(toPredict)
prediction

make="TESLA"
model="MODEL S P100D"
distance=353
curdist=150
res=elec.loc[(elec['make'] == make) & (elec['model'] == model)]
res

maxdist=res['dist'].astype('int')
maxdist

recharge=res['recharge'].astype('int')
recharge

year=res['year'].astype('int')
year

res['vclass'] = res['vclass'].replace('SUBCOMPACT', '1')
res['vclass'] = res['vclass'].replace('MID-SIZE', '2')
res['vclass'] = res['vclass'].replace('COMPACT', '3')
res['vclass'] = res['vclass'].replace('TWO-SEATER', '4')
res['vclass'] = res['vclass'].replace('FULL-SIZE', '5')
res['vclass'] = res['vclass'].replace('STATION WAGON - SMALL', '6')
res['vclass'] = res['vclass'].replace('SUV - STANDARD', '7')
vclass = res['vclass'].astype(int)
vclass

combGas=res['combGas'].astype('float')
combGas

combElec=res['combElec'].astype('float')
combElec

if (curdist>distance):
    rechargedist=0
    rechargetime=0;
    flag=0
else:
    rechargedist=distance-(curdist-16)
    rechargetime=recharge/maxdist*(rechargedist)
    flag=1
    
rechargetime

toPredict = pd.DataFrame({'vclass': vclass, 'year':year})
toPredict.head()

prediction=model.predict(toPredict)
prediction #shows co2 emitted in grams per km

co2=prediction*distance/1000 #in kg
co2

fuelconsgas=combGas/100*distance
fuelconsgas

fuelconselec=combElec/100*distance
fuelconselec

print("The time you need to reacharge your car to make your journey is %f "%rechargetime)
print("The fuel reqired for the journey is %f KWH and in Le is %f"%(fuelconselec,fuelconsgas))
print("Woo hoo you have saved %f kg of CO2 from destroying our earth."%co2)

import matplotlib.pyplot as plt
import numpy as np
dist=0
curdist=150
x=list()
y=list()
maxdist=507
recharge=12
while dist <1000:
    
    x.append(dist)
    if (curdist>dist):
        rechargetime=0
    else:
        #if(dist<maxdist):
        rechargedist=dist-(curdist-16)
        mul=rechargedist%maxdist
        add=int(rechargedist/maxdist)
        rechargetime=mul*recharge/maxdist + add*recharge 
    y.append(rechargetime)
    dist+=1;
plt.plot(x,y)
plt.show()

cols=['year','make','model','vclass','fuel','cityElec','hwyElec','combElec','cityGas','hwyGas','combGas','co2','co2Rate','smogRate','dist','recharge','co2saved']
db = pd.DataFrame(columns=cols)
db

for index, row in elec.iterrows():
    vclass=row["vclass"]
    if( vclass == 'SUBCOMPACT'):
        vclass=int(1)
    elif(vclass =='MID-SIZE'):
        vclass=int(2)
    elif(vclass =='COMPACT'):
        vclass=int(3)
    elif(vclass =='TWO-SEATER'):
        vclass=int(4)
    elif(vclass =='FULL-SIZE'):
        vclass=int(5)
    elif(vclass =='STATION WAGON - SMALL'):
        vclass=int(6)
    elif(vclass =='SUV - STANDARD'):
        vclass=int(7)
    year=int(row["year"])
    toPredict = pd.DataFrame({'vclass':[vclass], 'year':[year]})
    #print(toPredict.head())
    prediction=model.predict(toPredict)
    #print(prediction)
    dfrow=pd.DataFrame({'year':[year],'make':row["make"],'model':row["model"],'vclass':[vclass],'fuel':row["fuel"],'cityElec':row["cityElec"],'hwyElec':row["hwyElec"],'combElec':row["combElec"],'cityGas':row["cityGas"],'hwyGas':row["hwyGas"],'combGas':row["combGas"],'co2':row["co2"],'co2Rate':row["co2Rate"],'smogRate':row["smogRate"],'dist':row["dist"],'recharge':row["recharge"],'co2saved':prediction})
    db = db.append(dfrow)
db

db=db.reset_index(drop=True)
db

db.to_csv('Database.csv')
db.to_json('Databasejs.json')

db["vehicle"]=db["make"].astype(str) + "  "+db["model"].astype(str)
db

db.drop(["make","model"],inplace=True,axis=1)
db

db.to_csv('Database.csv')
db.to_json('Databasejs.json')

