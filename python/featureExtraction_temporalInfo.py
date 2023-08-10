import numpy as np
import pandas as pd
import datetime as dt

# loading original data
shopInfoFile = '../dataset/shop_info.txt'
shopInfo = pd.read_table(shopInfoFile, sep = ',', header = None)
shopInfo.columns = ['shopID', 'city', 'locationID', 'perPay', 'score', 'commentCnt', 'shopLevel', 'cate1', 'cate2', 'cate3']

columns = ['shopID', 'year', 'month', 'day', 'dayOfWeek', 'holiday', 'numHolidayLast', 'numHolidayCur', 'numHolidayNext']

startDateTrain = dt.date(2016, 9, 20)
endDateTrain = dt.date(2016, 10, 17)
startDateTest = dt.date(2016, 10, 18)
endDateTest = dt.date(2016, 10, 31)

startDayOfWeekTrain = 2
startDayOfWeekTest = 2
dayOfWeekTrain = []
dayOfWeekTest = []

cur = startDayOfWeekTrain
for i in range((endDateTrain - startDateTrain).days + 1):
    if cur > 7:
        cur = 1
    dayOfWeekTrain.append(cur)
    cur = cur + 1

cur = startDayOfWeekTest
for i in range((endDateTest - startDateTest).days + 1):
    if cur > 7:
        cur = 1
    dayOfWeekTest.append(cur)
    cur = cur + 1

numHolidayLastTrain = [2] * ((endDateTrain - startDateTrain).days + 1)
numHolidayLastTest = [2] * ((endDateTest - startDateTest).days + 1)

numHolidayLastTrain[0] = 3    #9-20
numHolidayLastTrain[1] = 3    #9-21
numHolidayLastTrain[2] = 3    #9-22
numHolidayLastTrain[3] = 2    #9-23
numHolidayLastTrain[4] = 1    #9-24
numHolidayLastTrain[5] = 1    #9-25

numHolidayLastTrain[14] = 3    #10-4
numHolidayLastTrain[15] = 4    #10-5
numHolidayLastTrain[16] = 5    #10-6
numHolidayLastTrain[17] = 6    #10-7
numHolidayLastTrain[18] = 7    #10-8
numHolidayLastTrain[19] = 6    #10-9
numHolidayLastTrain[20] = 5    #10-10
numHolidayLastTrain[21] = 4    #10-11
numHolidayLastTrain[22] = 3    #10-12
numHolidayLastTrain[23] = 2    #10-13
numHolidayLastTrain[24] = 1    #10-14
numHolidayLastTrain[25] = 0    #10-15
numHolidayLastTrain[26] = 1    #10-16

numHolidayCurTrain = [2] * ((endDateTrain - startDateTrain).days + 1)
numHolidayCurTest = [2] * ((endDateTest - startDateTest).days + 1)

numHolidayCurTrain[0] = 1    #9-20
numHolidayCurTrain[1] = 1    #9-21
numHolidayCurTrain[2] = 1    #9-22
numHolidayCurTrain[3] = 1    #9-23
numHolidayCurTrain[4] = 1    #9-24

numHolidayCurTrain[12] = 6    #10-2
numHolidayCurTrain[13] = 6    #10-3
numHolidayCurTrain[14] = 6    #10-4
numHolidayCurTrain[15] = 6    #10-5
numHolidayCurTrain[16] = 6    #10-6
numHolidayCurTrain[17] = 6    #10-7
numHolidayCurTrain[18] = 6    #10-8

numHolidayCurTrain[19] = 1    #10-9
numHolidayCurTrain[20] = 1    #10-10
numHolidayCurTrain[21] = 1    #10-11
numHolidayCurTrain[22] = 1    #10-12
numHolidayCurTrain[23] = 1    #10-13
numHolidayCurTrain[24] = 1    #10-14
numHolidayCurTrain[25] = 1    #10-15

numHolidayNextTrain = [2] * ((endDateTrain - startDateTrain).days + 1)
numHolidayNextTest = [2] * ((endDateTest - startDateTest).days + 1)

numHolidayNextTrain[6] = 3    #9-26
numHolidayNextTrain[7] = 4    #9-27
numHolidayNextTrain[8] = 5    #9-28
numHolidayNextTrain[9] = 6    #9-29
numHolidayNextTrain[10] = 7    #9-30
numHolidayNextTrain[11] = 6    #10-1
numHolidayNextTrain[12] = 5    #10-2
numHolidayNextTrain[13] = 4    #10-3
numHolidayNextTrain[14] = 3    #10-4
numHolidayNextTrain[15] = 2    #10-5
numHolidayNextTrain[16] = 1    #10-6
numHolidayNextTrain[17] = 0    #10-7
numHolidayNextTrain[18] = 1    #10-8

holidayTrain = []
for day in dayOfWeekTrain:
    if day < 6:
        holidayTrain.append(0)
    else:
        holidayTrain.append(1)

holidayTrain[11] = 2    #10-1
holidayTrain[12] = 2    #10-2
holidayTrain[13] = 2    #10-3
holidayTrain[14] = 2    #10-4
holidayTrain[15] = 2    #10-5
holidayTrain[16] = 2    #10-6
holidayTrain[17] = 2    #10-7
holidayTrain[18] = 0    #10-8
holidayTrain[19] = 0    #10-9
        
holidayTest = []
for day in dayOfWeekTest:
    if day < 6:
        holidayTest.append(0)
    else:
        holidayTest.append(1)

temporalDataTrain = {}
for column in columns:
    temporalDataTrain[column] = []

for shopID in shopInfo['shopID']:
    curDate = startDateTrain
    endDate = endDateTrain + dt.timedelta(days = 1)
    while curDate != endDate:
        temporalDataTrain['year'].append(curDate.year)
        temporalDataTrain['month'].append(curDate.month)
        temporalDataTrain['day'].append(curDate.day)
        temporalDataTrain['shopID'].append(shopID)
        temporalDataTrain['dayOfWeek'].append(dayOfWeekTrain[(curDate - startDateTrain).days])
        temporalDataTrain['holiday'].append(holidayTrain[(curDate - startDateTrain).days])
        temporalDataTrain['numHolidayLast'].append(numHolidayLastTrain[(curDate - startDateTrain).days])
        temporalDataTrain['numHolidayCur'].append(numHolidayCurTrain[(curDate - startDateTrain).days])
        temporalDataTrain['numHolidayNext'].append(numHolidayNextTrain[(curDate - startDateTrain).days])
        curDate = curDate + dt.timedelta(days = 1)

trainFeatures_temporalInfo = pd.DataFrame(temporalDataTrain, columns = columns)
trainFeatures_temporalInfo.to_csv('../preprocess/trainValidFeatures_temporalInfo.csv', header = False, index = False, date_format = 'int32')

temporalDataTest = {}
for column in columns:
    temporalDataTest[column] = []

for shopID in shopInfo['shopID']:
    curDate = startDateTest
    endDate = endDateTest + dt.timedelta(days = 1)
    while curDate != endDate:
        temporalDataTest['year'].append(curDate.year)
        temporalDataTest['month'].append(curDate.month)
        temporalDataTest['day'].append(curDate.day)
        temporalDataTest['shopID'].append(shopID)
        temporalDataTest['dayOfWeek'].append(dayOfWeekTest[(curDate - startDateTest).days])
        temporalDataTest['holiday'].append(holidayTest[(curDate - startDateTest).days])
        temporalDataTest['numHolidayLast'].append(numHolidayLastTest[(curDate - startDateTest).days])
        temporalDataTest['numHolidayCur'].append(numHolidayCurTest[(curDate - startDateTest).days])
        temporalDataTest['numHolidayNext'].append(numHolidayNextTest[(curDate - startDateTest).days])
        curDate = curDate + dt.timedelta(days = 1)

testFeatures_temporalInfo = pd.DataFrame(temporalDataTest, columns = columns)
testFeatures_temporalInfo.to_csv('../preprocess/validFeatures_temporalInfo.csv', header = False, index = False, date_format = 'int32')

startDateTrain = dt.date(2016, 10, 4)
endDateTrain = dt.date(2016, 10, 31)
startDateTest = dt.date(2016, 11, 1)
endDateTest = dt.date(2016, 11, 14)

startDayOfWeekTrain = 2
startDayOfWeekTest = 2
dayOfWeekTrain = []
dayOfWeekTest = []

cur = startDayOfWeekTrain
for i in range((endDateTrain - startDateTrain).days + 1):
    if cur > 7:
        cur = 1
    dayOfWeekTrain.append(cur)
    cur = cur + 1

cur = startDayOfWeekTest
for i in range((endDateTest - startDateTest).days + 1):
    if cur > 7:
        cur = 1
    dayOfWeekTest.append(cur)
    cur = cur + 1

numHolidayLastTrain = [2] * ((endDateTrain - startDateTrain).days + 1)
numHolidayLastTest = [2] * ((endDateTest - startDateTest).days + 1)

numHolidayLastTrain[0] = 3    #10-4
numHolidayLastTrain[1] = 4    #10-5
numHolidayLastTrain[2] = 5    #10-6
numHolidayLastTrain[3] = 6    #10-7
numHolidayLastTrain[4] = 7    #10-8
numHolidayLastTrain[5] = 6    #10-9
numHolidayLastTrain[6] = 5    #10-10
numHolidayLastTrain[7] = 4    #10-11
numHolidayLastTrain[8] = 3    #10-12
numHolidayLastTrain[9] = 2    #10-13
numHolidayLastTrain[10] = 1    #10-14
numHolidayLastTrain[11] = 0    #10-15
numHolidayLastTrain[12] = 1    #10-16

numHolidayCurTrain = [2] * ((endDateTrain - startDateTrain).days + 1)
numHolidayCurTest = [2] * ((endDateTest - startDateTest).days + 1)

numHolidayCurTrain[0] = 6    #10-4
numHolidayCurTrain[1] = 6    #10-5
numHolidayCurTrain[2] = 6    #10-6
numHolidayCurTrain[3] = 6    #10-7
numHolidayCurTrain[4] = 6    #10-8

numHolidayCurTrain[5] = 1    #10-9
numHolidayCurTrain[6] = 1    #10-10
numHolidayCurTrain[7] = 1    #10-11
numHolidayCurTrain[8] = 1    #10-12
numHolidayCurTrain[9] = 1    #10-13
numHolidayCurTrain[10] = 1    #10-14
numHolidayCurTrain[11] = 1    #10-15

numHolidayNextTrain = [2] * ((endDateTrain - startDateTrain).days + 1)
numHolidayNextTest = [2] * ((endDateTest - startDateTest).days + 1)

numHolidayNextTrain[0] = 3    #10-4
numHolidayNextTrain[1] = 2    #10-5
numHolidayNextTrain[2] = 1    #10-6
numHolidayNextTrain[3] = 0    #10-7
numHolidayNextTrain[4] = 1    #10-8

holidayTrain = []
for day in dayOfWeekTrain:
    if day < 6:
        holidayTrain.append(0)
    else:
        holidayTrain.append(1)

holidayTrain[0] = 2    #10-4
holidayTrain[1] = 2    #10-5
holidayTrain[2] = 2    #10-6
holidayTrain[3] = 2    #10-7
holidayTrain[4] = 0    #10-8
holidayTrain[5] = 0    #10-9

holidayTest = []
for day in dayOfWeekTest:
    if day < 6:
        holidayTest.append(0)
    else:
        holidayTest.append(1)

temporalDataTrain = {}
for column in columns:
    temporalDataTrain[column] = []

for shopID in shopInfo['shopID']:
    curDate = startDateTrain
    endDate = endDateTrain + dt.timedelta(days = 1)
    while curDate != endDate:
        temporalDataTrain['year'].append(curDate.year)
        temporalDataTrain['month'].append(curDate.month)
        temporalDataTrain['day'].append(curDate.day)
        temporalDataTrain['shopID'].append(shopID)
        temporalDataTrain['dayOfWeek'].append(dayOfWeekTrain[(curDate - startDateTrain).days])
        temporalDataTrain['holiday'].append(holidayTrain[(curDate - startDateTrain).days])
        temporalDataTrain['numHolidayLast'].append(numHolidayLastTrain[(curDate - startDateTrain).days])
        temporalDataTrain['numHolidayCur'].append(numHolidayCurTrain[(curDate - startDateTrain).days])
        temporalDataTrain['numHolidayNext'].append(numHolidayNextTrain[(curDate - startDateTrain).days])
        curDate = curDate + dt.timedelta(days = 1)

trainFeatures_temporalInfo = pd.DataFrame(temporalDataTrain, columns = columns)
trainFeatures_temporalInfo.to_csv('../preprocess/trainTestFeatures_temporalInfo.csv', header = False, index = False, date_format = 'int32')

temporalDataTest = {}
for column in columns:
    temporalDataTest[column] = []

for shopID in shopInfo['shopID']:
    curDate = startDateTest
    endDate = endDateTest + dt.timedelta(days = 1)
    while curDate != endDate:
        temporalDataTest['year'].append(curDate.year)
        temporalDataTest['month'].append(curDate.month)
        temporalDataTest['day'].append(curDate.day)
        temporalDataTest['shopID'].append(shopID)
        temporalDataTest['dayOfWeek'].append(dayOfWeekTest[(curDate - startDateTest).days])
        temporalDataTest['holiday'].append(holidayTest[(curDate - startDateTest).days])
        temporalDataTest['numHolidayLast'].append(numHolidayLastTest[(curDate - startDateTest).days])
        temporalDataTest['numHolidayCur'].append(numHolidayCurTest[(curDate - startDateTest).days])
        temporalDataTest['numHolidayNext'].append(numHolidayNextTest[(curDate - startDateTest).days])
        curDate = curDate + dt.timedelta(days = 1)

testFeatures_temporalInfo = pd.DataFrame(temporalDataTest, columns = columns)
testFeatures_temporalInfo.to_csv('../preprocess/testFeatures_temporalInfo.csv', header = False, index = False, date_format = 'int32')

