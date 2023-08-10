import pandas as pd
import numpy as np
import datetime as dt

# define basic date functions and weather function
def year(row):
    date = dt.datetime.strptime(row['date'], "%Y-%m-%d")
    return date.year

def month(row):
    date = dt.datetime.strptime(row['date'], "%Y-%m-%d")
    return date.month

def day(row):
    date = dt.datetime.strptime(row['date'], "%Y-%m-%d")
    return date.day

def change_date(date):
    date_list = date.split('/')
    return '{}-{:02}-{:02}'.format(int(date_list[0]), int(date_list[1]), int(date_list[2]))

def quantifyWeather(row):
    if row['weather'] in ['暴雨', '大到暴雨', '大雨', '大暴雨']:
        return 5
    elif row['weather'] in ['中雨', '中雪', '中到大雨', '中到大雪']:
        return 4
    elif row['weather'] in ['小到中雨', '雨夹雪', '小到中雪']:
        return 3
    elif row['weather'] in ['阵雨', '小雨', '阵雪', '雷阵雨', '小雪']:
        return 2
    elif row['weather'] in ['阴', '霾', '雾']:
        return 1
    elif row['weather'] in ['晴', '多云']:
        return 0

#loading original data
weather_info = pd.read_csv('../dataset/weather_info.csv')
pm = pd.read_csv('../dataset/pm2.5.csv')

#load time history data
payTH = np.loadtxt('../preprocess/payTH_parallel.txt', dtype = 'int32')

shopInfoFile = '../dataset/shop_info.txt'
shopInfo = pd.read_table(shopInfoFile, sep = ',', header = None)
shopInfo.columns = ['shopID', 'city', 'locationID', 'perPay', 'score', 'commentCnt', 'shopLevel', 'cate1', 'cate2', 'cate3']

# process the date information in the weather dataframe
weather_info.columns = ['city', 'date', 'maxTemp', 'minTemp', 'weather', 'wind_direction', 'wind_level']
weather_info = weather_info.drop(['wind_direction', 'wind_level'], axis=1)
weather_info['year'] = weather_info.apply (lambda row: year(row),axis=1)
weather_info['month'] = weather_info.apply (lambda row: month(row),axis=1)
weather_info['day'] = weather_info.apply (lambda row: day(row),axis=1)

# check weather information
weather_info = weather_info.drop(weather_info[weather_info['date'] > "2016-11-14"].index)
weather_info = weather_info.drop(weather_info[weather_info['date'] < "2016-09-20"].index)
weather_info['weather'].unique()

# generate weather code
weather_info.index = range(len(weather_info))
weather_info['weather'] = weather_info.apply (lambda row: quantifyWeather(row),axis=1)
pm['Date'] = list(map(change_date, pm['Date']))
pm_new = pm[pm['Date'] < "2016-11-15"]
pm_new = pm_new[pm_new['Date'] > "2016-09-19"]
pm_new.reset_index(drop=True, inplace=True)

startDateTrain = dt.date(2016, 9, 20)
endDateTrain = dt.date(2016, 10, 17)
startDateTest = dt.date(2016, 10, 18)
endDateTest = dt.date(2016, 10, 31)
startDate = dt.date(2015, 7, 1)
endDate = dt.date(2016, 10, 31)

columns = ['shopID', 'year', 'month', 'day', 'maxTemp', 'minTemp', 'weather', 'pm']

weatherDataTrain = {}
for column in columns:
    weatherDataTrain[column] = []

cities = shopInfo['city']
    
for shopID in shopInfo['shopID']:
    city = cities[shopID - 1]
    curDate = startDateTrain
    endDate = endDateTrain + dt.timedelta(days = 1)
    while curDate != endDate:
        cityWeather = weather_info[weather_info['city'] == city]
        weatherRecord = cityWeather[cityWeather['date'] == curDate.strftime('%Y-%m-%d')]
        
        for shopCol in columns:
            if shopCol == 'year':
                weatherDataTrain[shopCol].append(curDate.year)
            elif shopCol == 'month':
                weatherDataTrain[shopCol].append(curDate.month)
            elif shopCol == 'day':
                weatherDataTrain[shopCol].append(curDate.day)
            elif shopCol == 'shopID':
                weatherDataTrain[shopCol].append(shopID)
            elif shopCol == 'pm':
                pm = pm_new[pm_new['Date'] == curDate.strftime('%Y-%m-%d')][city].values[0]
                weatherDataTrain[shopCol].append(pm)
            else:
                value = weatherRecord[shopCol].values[0]
                weatherDataTrain[shopCol].append(value)
        curDate = curDate + dt.timedelta(days = 1)

trainFeatures_weather = pd.DataFrame(weatherDataTrain, columns = columns)
trainFeatures_weather.to_csv('../preprocess/trainValidFeatures_weather.csv', header = False, index = False, date_format = 'float32')

weatherDataTest = {}
for column in columns:
    weatherDataTest[column] = []

cities = shopInfo['city']
    
for shopID in shopInfo['shopID']:
    city = cities[shopID - 1]
    curDate = startDateTest
    endDate = endDateTest + dt.timedelta(days = 1)
    while curDate != endDate:
        cityWeather = weather_info[weather_info['city'] == city]
        weatherRecord = cityWeather[cityWeather['date'] == curDate.strftime('%Y-%m-%d')]
        
        for shopCol in columns:
            if shopCol == 'year':
                weatherDataTest[shopCol].append(curDate.year)
            elif shopCol == 'month':
                weatherDataTest[shopCol].append(curDate.month)
            elif shopCol == 'day':
                weatherDataTest[shopCol].append(curDate.day)
            elif shopCol == 'shopID':
                weatherDataTest[shopCol].append(shopID)
            elif shopCol == 'pm':
                pm = pm_new[pm_new['Date'] == curDate.strftime('%Y-%m-%d')][city].values[0]
                weatherDataTest[shopCol].append(pm)
            else:
                value = weatherRecord[shopCol].values[0]
                weatherDataTest[shopCol].append(value)
        curDate = curDate + dt.timedelta(days = 1)

testFeatures_weather = pd.DataFrame(weatherDataTest, columns = columns)
testFeatures_weather.to_csv('../preprocess/validFeatures_weather.csv', header = False, index = False, date_format = 'float32')

startDateTrain = dt.date(2016, 10, 4)
endDateTrain = dt.date(2016, 10, 31)
startDateTest = dt.date(2016, 11, 1)
endDateTest = dt.date(2016, 11, 14)
startDate = dt.date(2015, 7, 1)
endDate = dt.date(2016, 10, 31)

columns = ['shopID', 'year', 'month', 'day', 'maxTemp', 'minTemp', 'weather', 'pm']

weatherDataTrain = {}
for column in columns:
    weatherDataTrain[column] = []

cities = shopInfo['city']
    
for shopID in shopInfo['shopID']:
    city = cities[shopID - 1]
    curDate = startDateTrain
    endDate = endDateTrain + dt.timedelta(days = 1)
    while curDate != endDate:
        cityWeather = weather_info[weather_info['city'] == city]
        weatherRecord = cityWeather[cityWeather['date'] == curDate.strftime('%Y-%m-%d')]
        
        for shopCol in columns:
            if shopCol == 'year':
                weatherDataTrain[shopCol].append(curDate.year)
            elif shopCol == 'month':
                weatherDataTrain[shopCol].append(curDate.month)
            elif shopCol == 'day':
                weatherDataTrain[shopCol].append(curDate.day)
            elif shopCol == 'shopID':
                weatherDataTrain[shopCol].append(shopID)
            elif shopCol == 'pm':
                pm = pm_new[pm_new['Date'] == curDate.strftime('%Y-%m-%d')][city].values[0]
                weatherDataTrain[shopCol].append(pm)
            else:
                value = weatherRecord[shopCol].values[0]
                weatherDataTrain[shopCol].append(value)
        curDate = curDate + dt.timedelta(days = 1)

trainFeatures_weather = pd.DataFrame(weatherDataTrain, columns = columns)
trainFeatures_weather.to_csv('../preprocess/trainTestFeatures_weather.csv', header = False, index = False, date_format = 'float32')

weatherDataTest = {}
for column in columns:
    weatherDataTest[column] = []

cities = shopInfo['city']
    
for shopID in shopInfo['shopID']:
    city = cities[shopID - 1]
    curDate = startDateTest
    endDate = endDateTest + dt.timedelta(days = 1)
    while curDate != endDate:
        cityWeather = weather_info[weather_info['city'] == city]
        weatherRecord = cityWeather[cityWeather['date'] == curDate.strftime('%Y-%m-%d')]
        
        for shopCol in columns:
            if shopCol == 'year':
                weatherDataTest[shopCol].append(curDate.year)
            elif shopCol == 'month':
                weatherDataTest[shopCol].append(curDate.month)
            elif shopCol == 'day':
                weatherDataTest[shopCol].append(curDate.day)
            elif shopCol == 'shopID':
                weatherDataTest[shopCol].append(shopID)
            elif shopCol == 'pm':
                pm = pm_new[pm_new['Date'] == curDate.strftime('%Y-%m-%d')][city].values[0]
                weatherDataTest[shopCol].append(pm)
            else:
                value = weatherRecord[shopCol].values[0]
                weatherDataTest[shopCol].append(value)
        curDate = curDate + dt.timedelta(days = 1)

testFeatures_weather = pd.DataFrame(weatherDataTest, columns = columns)
testFeatures_weather.to_csv('../preprocess/testFeatures_weather.csv', header = False, index = False, date_format = 'float32')

