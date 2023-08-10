import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

properties = pd.read_csv('properties_2016.csv')
train = pd.read_csv('train_2016.csv')
sampleSub = pd.read_csv('sample_submission.csv')

properties = pd.read_csv('properties_2016.csv')

print properties['latitude'].value_counts()

xTrain = np.load('xTrain.npy')
yTrain = np.load('yTrain.npy')

xTrain.shape

train.head()

properties.tail()

sampleSub.head()

p = pd.read_csv('preds.csv')

p.head()

cleanedPropertyData = pd.np.array(properties)
cleanedPropertyData[0][0]

(cleanedPropertyData[0][0])

properties.describe()

xTrain.shape

properties[properties['parcelid'] == 10754147]

numTrainExamples = train.shape[0]
numFeatures = properties.shape[1]
xTrain = np.zeros([numTrainExamples, numFeatures])
yTrain = np.zeros([numTrainExamples])

propertiesIds = properties['parcelid']
rowCounter = 0
for index, row in train.iterrows():
    xTrain[rowCounter] = properties[properties['parcelid'] == row['parcelid']]
    yTrain[rowCounter] = row['logerror']
    rowCounter = rowCounter + 1
    if index == 10:
        break

type(properties.loc[0, 'parcelid'])

model = linear_model.LinearRegression()
model.fit(xTrain, yTrain)
preds = model.predict(properties)

cleanedPropertyData = pd.np.array(properties)
type((cleanedPropertyData[0][0]))

(cleanedPropertyData[0][0])

numTestExamples = properties.shape[0]
numPredictionColumns = 7
predictions = np.zeros([numTestExamples, numPredictionColumns])
for index, pred in enumerate(preds):
    predictions[index][0] = properties.loc[index, 'parcelid']
    predictions[index][1:7] = pred
    break
predictions[0]

sampleSub.head()

import csv
firstRow = [['ParcelId', '201610', '201611', '201612', '201710', '201711', '201712']]
with open("result.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(firstRow)
    writer.writerows(predictions)

airConditionMode = properties['airconditioningtypeid'].value_counts().argmax()
properties['airconditioningtypeid'] = properties['airconditioningtypeid'].fillna(airConditionMode)

architectureMode = properties['architecturalstyletypeid'].value_counts().argmax()
properties['architecturalstyletypeid'] = properties['architecturalstyletypeid'].fillna(architectureMode)

basementSqFeetAverage = properties['basementsqft'].mean()
properties['basementsqft'] = properties['basementsqft'].fillna(basementSqFeetAverage)

bathroomCntMode = properties['bathroomcnt'].value_counts().argmax()
properties['bathroomcnt'] = properties['bathroomcnt'].fillna(bathroomCntMode)

bedroomCntMode = properties['bedroomcnt'].value_counts().argmax()
properties['bedroomcnt'] = properties['bedroomcnt'].fillna(bedroomCntMode)

buildingClassType = properties['buildingclasstypeid'].value_counts().argmax()
properties['buildingclasstypeid'] = properties['buildingclasstypeid'].fillna(buildingClassType)

buildingQualityType = properties['buildingqualitytypeid'].value_counts().argmax()
properties['buildingqualitytypeid'] = properties['buildingqualitytypeid'].fillna(buildingQualityType)

calculatedBathnBedroom = properties['calculatedbathnbr'].value_counts().argmax()
properties['calculatedbathnbr'] = properties['calculatedbathnbr'].fillna(calculatedBathnBedroom)

# Making deck type a binary label
properties['decktypeid'] = properties['decktypeid'].fillna(0) 
properties['decktypeid'] = properties['decktypeid'].replace(66,1)

floor1SqFeetAverage = properties['finishedfloor1squarefeet'].mean()
properties['finishedfloor1squarefeet'] = properties['finishedfloor1squarefeet'].fillna(floor1SqFeetAverage)

calculatedSqFeetAverage = properties['calculatedfinishedsquarefeet'].mean()
properties['calculatedfinishedsquarefeet'] = properties['calculatedfinishedsquarefeet'].fillna(calculatedSqFeetAverage)

finishedSqFeet6 = properties['finishedsquarefeet6'].mean()
properties['finishedsquarefeet6'] = properties['finishedsquarefeet6'].fillna(finishedSqFeet6)

finishedSqFeet12 = properties['finishedsquarefeet12'].mean()
properties['finishedsquarefeet12'] = properties['finishedsquarefeet12'].fillna(finishedSqFeet12)

finishedSqFeet13 = properties['finishedsquarefeet13'].mean()
properties['finishedsquarefeet13'] = properties['finishedsquarefeet13'].fillna(finishedSqFeet13)

finishedSqFeet15 = properties['finishedsquarefeet15'].mean()
properties['finishedsquarefeet15'] = properties['finishedsquarefeet15'].fillna(finishedSqFeet15)

finishedSqFeet50 = properties['finishedsquarefeet50'].mean()
properties['finishedsquarefeet50'] = properties['finishedsquarefeet50'].fillna(finishedSqFeet50)

fips = properties['fips'].value_counts().argmax()
properties['fips'] = properties['fips'].fillna(fips)

# Making fireplace count\ a binary label
properties['fireplacecnt'] = properties['fireplacecnt'].replace([2,3,4,5,6,7,8,9],1)
properties['fireplacecnt'] = properties['fireplacecnt'].fillna(0) 

fullCntBathMode = properties['fullbathcnt'].value_counts().argmax()
properties['fullbathcnt'] = properties['fullbathcnt'].fillna(fullCntBathMode)

garageCntMode = properties['garagecarcnt'].value_counts().argmax()
properties['garagecarcnt'] = properties['garagecarcnt'].fillna(garageCntMode)

garageSqFeetMode = properties['garagetotalsqft'].value_counts().argmax()
properties['garagetotalsqft'] = properties['garagetotalsqft'].fillna(garageSqFeetMode)

# Making hot tub a binary label
properties['hashottuborspa'] = properties['hashottuborspa'].replace(True,1)
properties['hashottuborspa'] = properties['hashottuborspa'].fillna(0)

heatingMode = properties['heatingorsystemtypeid'].value_counts().argmax()
properties['heatingorsystemtypeid'] = properties['heatingorsystemtypeid'].fillna(heatingMode)

properties = properties.drop('latitude', axis=1)
properties = properties.drop('longitude', axis=1)

lotSizeMode = properties['lotsizesquarefeet'].value_counts().argmax()
properties['lotsizesquarefeet'] = properties['lotsizesquarefeet'].fillna(lotSizeMode)

# Making pool a binary label
properties['poolcnt'] = properties['poolcnt'].fillna(0)

properties['poolsizesum'] = properties['poolsizesum'].fillna(0)

# These properties show through with the previous features
properties = properties.drop('pooltypeid10', axis=1)
properties = properties.drop('pooltypeid2', axis=1)
properties = properties.drop('pooltypeid7', axis=1)

# Why would these even impact the price (but idk, maybe they're important)?
properties = properties.drop('propertycountylandusecode', axis=1)
properties = properties.drop('propertylandusetypeid', axis=1)
properties = properties.drop('propertyzoningdesc', axis=1)
properties = properties.drop('rawcensustractandblock', axis=1)
properties = properties.drop('censustractandblock', axis=1)

properties['regionidcounty'] = properties['regionidcounty'].replace([3101, 1286, 2061],[0,1,2])
properties['regionidcounty'] = properties['regionidcounty'].fillna(0) 

# No idea how to handle these features
properties = properties.drop('regionidcity', axis=1)
properties = properties.drop('regionidzip', axis=1)
properties = properties.drop('regionidneighborhood', axis=1)

# Don't bedroom and bathroom counts already do this?
properties = properties.drop('roomcnt', axis=1)
properties = properties.drop('threequarterbathnbr', axis=1)

# Making deck type a binary label
properties['storytypeid'] = properties['storytypeid'].fillna(0) 
properties['storytypeid'] = properties['storytypeid'].replace(7,1)

# Only has like a couple thousand non NA values, so not worth
properties = properties.drop('typeconstructiontypeid', axis=1)

unitMode = properties['unitcnt'].value_counts().argmax()
properties['unitcnt'] = properties['unitcnt'].fillna(unitMode)

yardSqFt17 = properties['yardbuildingsqft17'].mean()
properties['yardbuildingsqft17'] = properties['yardbuildingsqft17'].fillna(yardSqFt17)

yardSqFt26 = properties['yardbuildingsqft26'].mean()
properties['yardbuildingsqft26'] = properties['yardbuildingsqft26'].fillna(yardSqFt26)

yearBuilt = properties['yearbuilt'].mean()
properties['yearbuilt'] = properties['yearbuilt'].fillna(yearBuilt)

properties['numberofstories'] = properties['numberofstories'].fillna(0)

# Fireplace count already does this
properties = properties.drop('fireplaceflag', axis=1)

structureTax = properties['structuretaxvaluedollarcnt'].mean()
properties['structuretaxvaluedollarcnt'] = properties['structuretaxvaluedollarcnt'].fillna(structureTax)

landTax = properties['landtaxvaluedollarcnt'].mean()
properties['landtaxvaluedollarcnt'] = properties['landtaxvaluedollarcnt'].fillna(landTax)

tax = properties['taxamount'].mean()
properties['taxamount'] = properties['taxamount'].fillna(tax)

# Tax amount already does this
properties = properties.drop('taxvaluedollarcnt', axis=1)

# Idk, I don't wanna deal with these
properties = properties.drop('assessmentyear', axis=1)
properties = properties.drop('taxdelinquencyflag', axis=1)
properties = properties.drop('taxdelinquencyyear', axis=1)

properties.columns

cleanedPropertyData = pd.np.array(properties)
cleanedPropertyData.shape

cleanedPropertyData.nbytes

properties.info()

import csv
predictions = []
t = int(4.5)
y =9.4
predictions.append([t, y,y,y,y,y,y])
predictions.append([t, y,y,y,y,y,y])

firstRow = [['ParcelId', '201610', '201611', '201612', '201710', '201711', '201712']]
with open("temp.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(firstRow)
    writer.writerows(predictions)

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_boston
data, targets = load_boston(True)

allModels = {}

model = linear_model.BayesianRidge()
predicted = cross_val_score(model, data, targets, scoring='neg_mean_absolute_error', cv=10)
allModels[model] = predicted.mean()

model = linear_model.LinearRegression()
predicted = cross_val_score(model, data, targets, scoring='neg_mean_absolute_error', cv=10)
allModels[model] = predicted.mean()

allModels



