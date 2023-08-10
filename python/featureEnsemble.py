import os
import pandas as pd

# loading preprocessed data and 
fileNames = os.listdir('../preprocess')
filePathPrefix = '../preprocess/'

periods = [7, 14, 28, 56, 112]
stats = ['mean', 'std', 'skew', 'kurtosis']
recentDataColumns = []
for period in periods:
    for stat in stats:
        column  = 'last' + str(period) + 'days_' + stat
        recentDataColumns.append(column)

periods = [7, 14, 28]
stats = ['meanView', 'stdView', 'skewView', 'kurtosisView']
recentDataViewColumns = []
for period in periods:
    for stat in stats:
        column = 'last' + str(period) + 'days_' + stat
        recentDataViewColumns.append(column)        
        
periods = [7, 14, 28, 56, 112]
trends = ['copy', 'ridge']
currentTrendcolumns = []
for period in periods:
    for trend in trends:
        column = 'last' + str(period) + 'days_' + trend
        currentTrendcolumns.append(column)
        
primaryKey = ['shopID', 'year', 'month', 'day']
columnDic = {
    'basicInfo':['city', 'perPay', 'score', 'commentCnt', 'shopLevel', 'category'],
    'recentData':recentDataColumns,
    'recentDataView':recentDataViewColumns,
    'currentTrend':currentTrendcolumns,
    'temporalInfo':['dayOfWeek', 'holiday', 'numHolidayLast', 'numHolidayCur', 'numHolidayNext'],
    'weather':['maxTemp', 'minTemp', 'weather', 'pm']
}

ensembleCol = ['shopID', 'year', 'month', 'day']
orderCol = ['basicInfo', 'recentData', 'temporalInfo', 'currentTrend', 'weather', 'recentDataView']
for col in orderCol:
    ensembleCol = ensembleCol + columnDic[col]

# loading file directories
trainFeatureFiles = []
testFeatureFiles = []
featureTypeTrain = []
featureTypeTest = []

for file in fileNames:
    if 'validFeatures' in file:
        featureType = file.split('_')[1].split('.')[0]
        testFeatureFiles.append(file)
        featureTypeTest.append(featureType)
    if 'trainValidFeatures' in file:
        featureType = file.split('_')[1].split('.')[0]
        trainFeatureFiles.append(file)
        featureTypeTrain.append(featureType)

ensembleTrain = pd.read_csv(filePathPrefix + trainFeatureFiles[0], header = None)
ensembleTrain.columns = primaryKey + columnDic[featureTypeTrain[0]]
ensembleTrain = ensembleTrain[primaryKey]

for index, file in enumerate(trainFeatureFiles):
    columns = columnDic[featureTypeTrain[index]]
    file_load = pd.read_csv(filePathPrefix + trainFeatureFiles[index], header = None)
    file_load.columns = primaryKey + columns
    ensembleTrain[columns] = file_load[columns]

ensembleTrain = ensembleTrain[ensembleCol]
ensembleTrain.to_csv('../preprocess/trainValidFeatures_ensemble.csv', header = False, index = False)

ensembleTest = pd.read_csv(filePathPrefix + testFeatureFiles[0], header = None)
ensembleTest.columns = primaryKey + columnDic[featureTypeTest[0]]
ensembleTest = ensembleTest[primaryKey]

for index, file in enumerate(testFeatureFiles):
    columns = columnDic[featureTypeTest[index]]
    file_load = pd.read_csv(filePathPrefix + testFeatureFiles[index], header = None)
    file_load.columns = primaryKey + columns
    ensembleTest[columns] = file_load[columns]

ensembleTest = ensembleTest[ensembleCol]
ensembleTest.to_csv('../preprocess/validFeatures_ensemble.csv', header = False, index = False)

ensembleTrain

ensembleTest

trainFeatureFiles = []
testFeatureFiles = []
featureTypeTrain = []
featureTypeTest = []

for file in fileNames:
    if 'testFeatures' in file:
        featureType = file.split('_')[1].split('.')[0]
        testFeatureFiles.append(file)
        featureTypeTest.append(featureType)
    if 'trainTestFeatures' in file:
        featureType = file.split('_')[1].split('.')[0]
        trainFeatureFiles.append(file)
        featureTypeTrain.append(featureType)

ensembleTrain = pd.read_csv(filePathPrefix + trainFeatureFiles[0], header = None)
ensembleTrain.columns = primaryKey + columnDic[featureTypeTrain[0]]
ensembleTrain = ensembleTrain[primaryKey]

for index, file in enumerate(trainFeatureFiles):
    columns = columnDic[featureTypeTrain[index]]
    file_load = pd.read_csv(filePathPrefix + trainFeatureFiles[index], header = None)
    file_load.columns = primaryKey + columns
    ensembleTrain[columns] = file_load[columns]

ensembleTrain = ensembleTrain[ensembleCol]
ensembleTrain.to_csv('../preprocess/trainTestFeatures_ensemble.csv', header = False, index = False)

ensembleTest = pd.read_csv(filePathPrefix + testFeatureFiles[0], header = None)
ensembleTest.columns = primaryKey + columnDic[featureTypeTest[0]]
ensembleTest = ensembleTest[primaryKey]

for index, file in enumerate(testFeatureFiles):
    columns = columnDic[featureTypeTest[index]]
    file_load = pd.read_csv(filePathPrefix + testFeatureFiles[index], header = None)
    file_load.columns = primaryKey + columns
    ensembleTest[columns] = file_load[columns]

ensembleTest = ensembleTest[ensembleCol]
ensembleTest.to_csv('../preprocess/testFeatures_ensemble.csv', header = False, index = False)

ensembleTrain

ensembleTest

