import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as st
from sklearn.linear_model import Ridge

#load time history data
payTH = np.loadtxt('../preprocess/payTH_parallel.txt', dtype = 'int32')

shopInfoFile = '../dataset/shop_info.txt'
shopInfo = pd.read_table(shopInfoFile, sep = ',', header = None)
shopInfo.columns = ['shopID', 'city', 'locationID', 'perPay', 'score', 'commentCnt', 'shopLevel', 'cate1', 'cate2', 'cate3']

# prepare current trend data
startDateTrain = dt.date(2016, 9, 20)
endDateTrain = dt.date(2016, 10, 17)
startDateTest = dt.date(2016, 10, 18)
endDateTest = dt.date(2016, 10, 31)
startDate = dt.date(2015, 7, 1)
endDate = dt.date(2016, 10, 31)

periods = [7, 14, 28, 56, 112]
trends = ['copy', 'ridge']
columns = ['shopID', 'year', 'month', 'day']
for period in periods:
    for trend in trends:
        column = 'last' + str(period) + 'days_' + trend
        columns.append(column)

dayNumTrain = []
dayNumTest = []
startDayNumTrain = (startDateTrain - startDate).days
startDayNumTest = (startDateTest - startDate).days
for period in periods:
    dayNumTrain.append(np.arange(startDayNumTrain - period, startDayNumTrain))
    dayNumTest.append(np.arange(startDayNumTest - period, startDayNumTest))

patternDayOfWeekTrain = [2, 3, 4, 5, 6, 7, 1]
patternDayOfWeekTest = [2, 3, 4, 5, 6, 7, 1]
dayOfWeekTrain = []
dayOfWeekTest = []
for period in periods:
    repeat = int(period/7)
    dayOfWeekTrain.append(np.array(patternDayOfWeekTrain * repeat))
    dayOfWeekTest.append(np.array(patternDayOfWeekTest * repeat))

trendListTrain = []
for index, pay in enumerate(payTH):
    trendListTrain.append([])
    days = (endDateTrain - startDateTrain).days + 1
    for i in range(days):
        trendListTrain[index].append([])
        for j, period in enumerate(periods):
            cur = (startDateTrain - startDate).days + i
            end = cur - i - 1
            start = end - period + 1
            dataCal = pay[start:(end+1)]
    
            curDayOfWeek = patternDayOfWeekTrain[i%7]
            dataCopy = dataCal[dayOfWeekTrain[j] == curDayOfWeek]
            if dataCopy[dataCopy != 0].size > 0:
                copy = np.mean(dataCopy[dataCopy != 0])
            else:
                copy = np.nan
            
            if dataCal[dataCal != 0].size > 0:
                y = dataCal[dataCal != 0]
                X = np.array([dayNumTrain[j][dataCal != 0], dayOfWeekTrain[j][dataCal != 0]]).T
                clf = Ridge(alpha=1.0)
                clf.fit(X, y)
                curX = np.array([cur, curDayOfWeek]).reshape(1, -1)
                ridge = clf.predict(curX)[0]
            else:
                ridge = np.nan
                
            trendListTrain[index][i].append([copy, ridge])

trendListTest = []
for index, pay in enumerate(payTH):
    trendListTest.append([])
    days = (endDateTest - startDateTest).days + 1
    for i in range(days):
        trendListTest[index].append([])
        for j, period in enumerate(periods):
            cur = (startDateTest - startDate).days + i
            end = cur - i - 1
            start = end - period + 1
            dataCal = pay[start:(end+1)]
    
            curDayOfWeek = patternDayOfWeekTest[i%7]
            dataCopy = dataCal[dayOfWeekTest[j] == curDayOfWeek]
            if dataCopy[dataCopy != 0].size > 0:
                copy = np.mean(dataCopy[dataCopy != 0])
            else:
                copy = np.nan
            
            if dataCal[dataCal != 0].size > 0:
                y = dataCal[dataCal != 0]
                X = np.array([dayNumTest[j][dataCal != 0], dayOfWeekTest[j][dataCal != 0]]).T
                clf = Ridge(alpha=1.0)
                clf.fit(X, y)
                curX = np.array([cur, curDayOfWeek]).reshape(1, -1)
                ridge = clf.predict(curX)[0]
            else:
                ridge = np.nan
                
            trendListTest[index][i].append([copy, ridge])

trendDataTrain = {}
for column in columns:
    trendDataTrain[column] = []

for shopID in shopInfo['shopID']:
    curDate = startDateTrain
    endDate = endDateTrain + dt.timedelta(days = 1)
    while curDate != endDate:
        for shopCol in columns:
            if shopCol == 'year':
                trendDataTrain[shopCol].append(curDate.year)
            elif shopCol == 'month':
                trendDataTrain[shopCol].append(curDate.month)
            elif shopCol == 'day':
                trendDataTrain[shopCol].append(curDate.day)
            elif shopCol == 'shopID':
                trendDataTrain[shopCol].append(shopID)
            else:
                period = int(shopCol.split('days')[0].split('last')[1])
                trend = shopCol.split('_')[-1]
                indexPeriod = periods.index(period)
                indexTrend = trends.index(trend)
                indexDate = (curDate - startDateTrain).days
#                 print(shopID, indexDate, indexPeriod, indexTrend)
                trendDataTrain[shopCol].append(trendListTrain[shopID - 1][indexDate][indexPeriod][indexTrend])            
        curDate = curDate + dt.timedelta(days = 1)
        
trainFeatures_currentTrend = pd.DataFrame(trendDataTrain, columns = columns)

trainFeatures_currentTrend = pd.DataFrame(trendDataTrain, columns = columns)
trainFeatures_currentTrend.to_csv('../preprocess/trainValidFeatures_currentTrend.csv', header = False, index = False, date_format = 'float32')

trendDataTest = {}
for column in columns:
    trendDataTest[column] = []

for shopID in shopInfo['shopID']:
    curDate = startDateTest
    endDate = endDateTest + dt.timedelta(days = 1)
    while curDate != endDate:
        for shopCol in columns:
            if shopCol == 'year':
                trendDataTest[shopCol].append(curDate.year)
            elif shopCol == 'month':
                trendDataTest[shopCol].append(curDate.month)
            elif shopCol == 'day':
                trendDataTest[shopCol].append(curDate.day)
            elif shopCol == 'shopID':
                trendDataTest[shopCol].append(shopID)
            else:
                period = int(shopCol.split('days')[0].split('last')[1])
                trend = shopCol.split('_')[-1]
                indexPeriod = periods.index(period)
                indexTrend = trends.index(trend)
                indexDate = (curDate - startDateTest).days
#                 print(shopID, indexDate, indexPeriod, indexTrend)
                trendDataTest[shopCol].append(trendListTest[shopID - 1][indexDate][indexPeriod][indexTrend])            
        curDate = curDate + dt.timedelta(days = 1)
        
testFeatures_currentTrend = pd.DataFrame(trendDataTest, columns = columns)

testFeatures_currentTrend = pd.DataFrame(trendDataTest, columns = columns)
testFeatures_currentTrend.to_csv('../preprocess/validFeatures_currentTrend.csv', header = False, index = False, date_format = 'float32')

startDateTrain = dt.date(2016, 10, 4)
endDateTrain = dt.date(2016, 10, 31)
startDateTest = dt.date(2016, 11, 1)
endDateTest = dt.date(2016, 11, 14)
startDate = dt.date(2015, 7, 1)
endDate = dt.date(2016, 10, 31)

periods = [7, 14, 28, 56, 112]
trends = ['copy', 'ridge']
columns = ['shopID', 'year', 'month', 'day']
for period in periods:
    for trend in trends:
        column = 'last' + str(period) + 'days_' + trend
        columns.append(column)

dayNumTrain = []
dayNumTest = []
startDayNumTrain = (startDateTrain - startDate).days
startDayNumTest = (startDateTest - startDate).days
for period in periods:
    dayNumTrain.append(np.arange(startDayNumTrain - period, startDayNumTrain))
    dayNumTest.append(np.arange(startDayNumTest - period, startDayNumTest))

patternDayOfWeekTrain = [2, 3, 4, 5, 6, 7, 1]
patternDayOfWeekTest = [2, 3, 4, 5, 6, 7, 1]
dayOfWeekTrain = []
dayOfWeekTest = []
for period in periods:
    repeat = int(period/7)
    dayOfWeekTrain.append(np.array(patternDayOfWeekTrain * repeat))
    dayOfWeekTest.append(np.array(patternDayOfWeekTest * repeat))

trendListTrain = []
for index, pay in enumerate(payTH):
    trendListTrain.append([])
    days = (endDateTrain - startDateTrain).days + 1
    for i in range(days):
        trendListTrain[index].append([])
        for j, period in enumerate(periods):
            cur = (startDateTrain - startDate).days + i
            end = cur - i - 1
            start = end - period + 1
            dataCal = pay[start:(end+1)]
    
            curDayOfWeek = patternDayOfWeekTrain[i%7]
            dataCopy = dataCal[dayOfWeekTrain[j] == curDayOfWeek]
            if dataCopy[dataCopy != 0].size > 0:
                copy = np.mean(dataCopy[dataCopy != 0])
            else:
                copy = np.nan
            
            if dataCal[dataCal != 0].size > 0:
                y = dataCal[dataCal != 0]
                X = np.array([dayNumTrain[j][dataCal != 0], dayOfWeekTrain[j][dataCal != 0]]).T
                clf = Ridge(alpha=1.0)
                clf.fit(X, y)
                curX = np.array([cur, curDayOfWeek]).reshape(1, -1)
                ridge = clf.predict(curX)[0]
            else:
                ridge = np.nan
                
            trendListTrain[index][i].append([copy, ridge])

trendListTest = []
for index, pay in enumerate(payTH):
    trendListTest.append([])
    days = (endDateTest - startDateTest).days + 1
    for i in range(days):
        trendListTest[index].append([])
        for j, period in enumerate(periods):
            cur = (startDateTest - startDate).days + i
            end = cur - i - 1
            start = end - period + 1
            dataCal = pay[start:(end+1)]
    
            curDayOfWeek = patternDayOfWeekTest[i%7]
            dataCopy = dataCal[dayOfWeekTest[j] == curDayOfWeek]
            if dataCopy[dataCopy != 0].size > 0:
                copy = np.mean(dataCopy[dataCopy != 0])
            else:
                copy = np.nan
            
            if dataCal[dataCal != 0].size > 0:
                y = dataCal[dataCal != 0]
                X = np.array([dayNumTest[j][dataCal != 0], dayOfWeekTest[j][dataCal != 0]]).T
                clf = Ridge(alpha=1.0)
                clf.fit(X, y)
                curX = np.array([cur, curDayOfWeek]).reshape(1, -1)
                ridge = clf.predict(curX)[0]
            else:
                ridge = np.nan
                
            trendListTest[index][i].append([copy, ridge])

trendDataTrain = {}
for column in columns:
    trendDataTrain[column] = []

for shopID in shopInfo['shopID']:
    curDate = startDateTrain
    endDate = endDateTrain + dt.timedelta(days = 1)
    while curDate != endDate:
        for shopCol in columns:
            if shopCol == 'year':
                trendDataTrain[shopCol].append(curDate.year)
            elif shopCol == 'month':
                trendDataTrain[shopCol].append(curDate.month)
            elif shopCol == 'day':
                trendDataTrain[shopCol].append(curDate.day)
            elif shopCol == 'shopID':
                trendDataTrain[shopCol].append(shopID)
            else:
                period = int(shopCol.split('days')[0].split('last')[1])
                trend = shopCol.split('_')[-1]
                indexPeriod = periods.index(period)
                indexTrend = trends.index(trend)
                indexDate = (curDate - startDateTrain).days
#                 print(shopID, indexDate, indexPeriod, indexTrend)
                trendDataTrain[shopCol].append(trendListTrain[shopID - 1][indexDate][indexPeriod][indexTrend])            
        curDate = curDate + dt.timedelta(days = 1)
        
trainFeatures_currentTrend = pd.DataFrame(trendDataTrain, columns = columns)

trainFeatures_currentTrend = pd.DataFrame(trendDataTrain, columns = columns)
trainFeatures_currentTrend.to_csv('../preprocess/trainTestFeatures_currentTrend.csv', header = False, index = False, date_format = 'float32')

trendDataTest = {}
for column in columns:
    trendDataTest[column] = []

for shopID in shopInfo['shopID']:
    curDate = startDateTest
    endDate = endDateTest + dt.timedelta(days = 1)
    while curDate != endDate:
        for shopCol in columns:
            if shopCol == 'year':
                trendDataTest[shopCol].append(curDate.year)
            elif shopCol == 'month':
                trendDataTest[shopCol].append(curDate.month)
            elif shopCol == 'day':
                trendDataTest[shopCol].append(curDate.day)
            elif shopCol == 'shopID':
                trendDataTest[shopCol].append(shopID)
            else:
                period = int(shopCol.split('days')[0].split('last')[1])
                trend = shopCol.split('_')[-1]
                indexPeriod = periods.index(period)
                indexTrend = trends.index(trend)
                indexDate = (curDate - startDateTest).days
#                 print(shopID, indexDate, indexPeriod, indexTrend)
                trendDataTest[shopCol].append(trendListTest[shopID - 1][indexDate][indexPeriod][indexTrend])            
        curDate = curDate + dt.timedelta(days = 1)
        
testFeatures_currentTrend = pd.DataFrame(trendDataTest, columns = columns)

testFeatures_currentTrend = pd.DataFrame(trendDataTest, columns = columns)
testFeatures_currentTrend.to_csv('../preprocess/testFeatures_currentTrend.csv', header = False, index = False, date_format = 'float32')

