import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as st

#load time history data
viewTH = np.loadtxt('../preprocess/viewTH_parallel.txt', dtype = 'int32')

shopInfoFile = '../dataset/shop_info.txt'
shopInfo = pd.read_table(shopInfoFile, sep = ',', header = None)
shopInfo.columns = ['shopID', 'city', 'locationID', 'perPay', 'score', 'commentCnt', 'shopLevel', 'cate1', 'cate2', 'cate3']

startDateTrain = dt.date(2016, 9, 20)
endDateTrain = dt.date(2016, 10, 17)
startDateTest = dt.date(2016, 10, 18)
endDateTest = dt.date(2016, 10, 31)
startDate = dt.date(2015, 7, 1)
endDate = dt.date(2016, 10, 31)

periods = [7, 14, 28]
stats = ['meanView', 'stdView', 'skewView', 'kurtosisView']
columns = ['shopID', 'year', 'month', 'day']
for period in periods:
    for stat in stats:
        column = 'last' + str(period) + 'days_' + stat
        columns.append(column)

statListTrain = []
for index, pay in enumerate(viewTH):
    statListTrain.append([])
    days = (endDateTrain - startDateTrain).days + 1
    for i in range(days):
        statListTrain[index].append([])
        for j, period in enumerate(periods):
            cur = (startDateTrain - startDate).days + i
            end = cur - days
            start = end - period + 1
            dataCal = pay[start:(end+1)]
            dataCal = dataCal[dataCal != 0]   #remove zero values
            
            if dataCal.size > period/2:
                mean = np.mean(dataCal)
                std = np.std(dataCal)
                skew = st.skew(dataCal)
                kurtosis = st.kurtosis(dataCal)
            else:
                mean = np.nan
                std = np.nan
                skew = np.nan
                kurtosis = np.nan
                
            statListTrain[index][i].append([mean, std, skew, kurtosis])

statListTest = []
for index, pay in enumerate(viewTH):
    statListTest.append([])
    days = (endDateTest - startDateTest).days + 1
    for i in range(days):
        statListTest[index].append([])
        for j, period in enumerate(periods):
            cur = (startDateTest - startDate).days + i
            end = cur - days
            start = end - period + 1
            dataCal = pay[start:(end+1)]
            dataCal = dataCal[dataCal != 0]   #remove zero values
        
            if dataCal.size > period/2:
                mean = np.mean(dataCal)
                std = np.std(dataCal)
                skew = st.skew(dataCal)
                kurtosis = st.kurtosis(dataCal)
            else:
                mean = np.nan
                std = np.nan
                skew = np.nan
                kurtosis = np.nan
            
            statListTest[index][i].append([mean, std, skew, kurtosis])

statDataTrain = {}
for column in columns:
    statDataTrain[column] = []

for shopID in shopInfo['shopID']:
    curDate = startDateTrain
    endDate = endDateTrain + dt.timedelta(days = 1)
    while curDate != endDate:
        for shopCol in columns:
            if shopCol == 'year':
                statDataTrain[shopCol].append(curDate.year)
            elif shopCol == 'month':
                statDataTrain[shopCol].append(curDate.month)
            elif shopCol == 'day':
                statDataTrain[shopCol].append(curDate.day)
            elif shopCol == 'shopID':
                statDataTrain[shopCol].append(shopID)
            else:
                period = int(shopCol.split('days')[0].split('last')[1])
                stat = shopCol.split('_')[-1]
                indexPeriod = periods.index(period)
                indexStat = stats.index(stat)
                indexDate = (curDate - startDateTrain).days
#                 print(shopID, indexDate, indexPeriod, indexStat)
                statDataTrain[shopCol].append(statListTrain[shopID - 1][indexDate][indexPeriod][indexStat])            
        curDate = curDate + dt.timedelta(days = 1)

trainFeatures_recentData = pd.DataFrame(statDataTrain, columns = columns)
trainFeatures_recentData.to_csv('../preprocess/trainValidFeatures_recentDataView.csv', header = False, index = False, date_format = 'float32')

statDataTest = {}
for column in columns:
    statDataTest[column] = []

for shopID in shopInfo['shopID']:
    curDate = startDateTest
    endDate = endDateTest + dt.timedelta(days = 1)
    while curDate != endDate:
        for shopCol in columns:
            if shopCol == 'year':
                statDataTest[shopCol].append(curDate.year)
            elif shopCol == 'month':
                statDataTest[shopCol].append(curDate.month)
            elif shopCol == 'day':
                statDataTest[shopCol].append(curDate.day)
            elif shopCol == 'shopID':
                statDataTest[shopCol].append(shopID)
            else:
                period = int(shopCol.split('days')[0].split('last')[1])
                stat = shopCol.split('_')[-1]
                indexPeriod = periods.index(period)
                indexStat = stats.index(stat)
                indexDate = (curDate - startDateTest).days
#                 print(shopID, indexDate, indexPeriod, indexStat)
                statDataTest[shopCol].append(statListTest[shopID - 1][indexDate][indexPeriod][indexStat])            
        curDate = curDate + dt.timedelta(days = 1)

testFeatures_recentData = pd.DataFrame(statDataTest, columns = columns)
testFeatures_recentData.to_csv('../preprocess/validFeatures_recentDataView.csv', header = False, index = False, date_format = 'float32')

startDateTrain = dt.date(2016, 10, 4)
endDateTrain = dt.date(2016, 10, 31)
startDateTest = dt.date(2016, 11, 1)
endDateTest = dt.date(2016, 11, 14)
startDate = dt.date(2015, 7, 1)
endDate = dt.date(2016, 10, 31)

periods = [7, 14, 28]
stats = ['meanView', 'stdView', 'skewView', 'kurtosisView']
columns = ['shopID', 'year', 'month', 'day']
for period in periods:
    for stat in stats:
        column = 'last' + str(period) + 'days_' + stat
        columns.append(column)

statListTrain = []
for index, pay in enumerate(viewTH):
    statListTrain.append([])
    days = (endDateTrain - startDateTrain).days + 1
    for i in range(days):
        statListTrain[index].append([])
        for j, period in enumerate(periods):
            cur = (startDateTrain - startDate).days + i
            end = cur - days
            start = end - period + 1
            dataCal = pay[start:(end+1)]
            dataCal = dataCal[dataCal != 0]   #remove zero values
            
            if dataCal.size > period/2:
                mean = np.mean(dataCal)
                std = np.std(dataCal)
                skew = st.skew(dataCal)
                kurtosis = st.kurtosis(dataCal)
            else:
                mean = np.nan
                std = np.nan
                skew = np.nan
                kurtosis = np.nan
                
            statListTrain[index][i].append([mean, std, skew, kurtosis])

statListTest = []
for index, pay in enumerate(viewTH):
    statListTest.append([])
    days = (endDateTest - startDateTest).days + 1
    for i in range(days):
        statListTest[index].append([])
        for j, period in enumerate(periods):
            cur = (startDateTest - startDate).days + i
            end = cur - days
            start = end - period + 1
            dataCal = pay[start:(end+1)]
            dataCal = dataCal[dataCal != 0]   #remove zero values
        
            if dataCal.size > period/2:
                mean = np.mean(dataCal)
                std = np.std(dataCal)
                skew = st.skew(dataCal)
                kurtosis = st.kurtosis(dataCal)
            else:
                mean = np.nan
                std = np.nan
                skew = np.nan
                kurtosis = np.nan
            
            statListTest[index][i].append([mean, std, skew, kurtosis])

statDataTrain = {}
for column in columns:
    statDataTrain[column] = []

for shopID in shopInfo['shopID']:
    curDate = startDateTrain
    endDate = endDateTrain + dt.timedelta(days = 1)
    while curDate != endDate:
        for shopCol in columns:
            if shopCol == 'year':
                statDataTrain[shopCol].append(curDate.year)
            elif shopCol == 'month':
                statDataTrain[shopCol].append(curDate.month)
            elif shopCol == 'day':
                statDataTrain[shopCol].append(curDate.day)
            elif shopCol == 'shopID':
                statDataTrain[shopCol].append(shopID)
            else:
                period = int(shopCol.split('days')[0].split('last')[1])
                stat = shopCol.split('_')[-1]
                indexPeriod = periods.index(period)
                indexStat = stats.index(stat)
                indexDate = (curDate - startDateTrain).days
#                 print(shopID, indexDate, indexPeriod, indexStat)
                statDataTrain[shopCol].append(statListTrain[shopID - 1][indexDate][indexPeriod][indexStat])            
        curDate = curDate + dt.timedelta(days = 1)

trainFeatures_recentData = pd.DataFrame(statDataTrain, columns = columns)
trainFeatures_recentData.to_csv('../preprocess/trainTestFeatures_recentDataView.csv', header = False, index = False, date_format = 'float32')

statDataTest = {}
for column in columns:
    statDataTest[column] = []

for shopID in shopInfo['shopID']:
    curDate = startDateTest
    endDate = endDateTest + dt.timedelta(days = 1)
    while curDate != endDate:
        for shopCol in columns:
            if shopCol == 'year':
                statDataTest[shopCol].append(curDate.year)
            elif shopCol == 'month':
                statDataTest[shopCol].append(curDate.month)
            elif shopCol == 'day':
                statDataTest[shopCol].append(curDate.day)
            elif shopCol == 'shopID':
                statDataTest[shopCol].append(shopID)
            else:
                period = int(shopCol.split('days')[0].split('last')[1])
                stat = shopCol.split('_')[-1]
                indexPeriod = periods.index(period)
                indexStat = stats.index(stat)
                indexDate = (curDate - startDateTest).days
#                 print(shopID, indexDate, indexPeriod, indexStat)
                statDataTest[shopCol].append(statListTest[shopID - 1][indexDate][indexPeriod][indexStat])            
        curDate = curDate + dt.timedelta(days = 1)

testFeatures_recentData = pd.DataFrame(statDataTest, columns = columns)
testFeatures_recentData.to_csv('../preprocess/testFeatures_recentDataView.csv', header = False, index = False, date_format = 'float32')

