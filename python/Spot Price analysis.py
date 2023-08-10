# Importing the necessary packages.

import boto3
import pandas as pd
import datetime

def handler(event, context):
    start_time = event['start_time']
    end_time = event ['end_time']
    region = event['region']
    product_description = event['product_description']
    client = boto3.client('ec2', region_name=region)
    response = client.describe_spot_price_history(
        InstanceTypes=event['instances_list'],
        ProductDescriptions=product_description,
        StartTime=start_time,
        EndTime = end_time,
        MaxResults=10000
    )
    return response['SpotPriceHistory']

def wrapper(instanceList, ProductDescriptionList, region):
    m4_list = []
    for i in range(1,90):
        output = (handler({
        'instances_list': instanceList,
        'start_time': datetime.datetime.now() - datetime.timedelta(i),
        'end_time': datetime.datetime.now() - datetime.timedelta(i-1),
        'product_description': ProductDescriptionList,
        'region': region
    }, ''))
        for j in range(0,len(output)):
            m4_list.append(output[j])

    df = pd.DataFrame(m4_list)
    df = df.drop_duplicates()
    df.reset_index(drop=True,inplace=True)
    return df

df = wrapper(['m4.large', 'm4.xlarge'],['Linux/UNIX (Amazon VPC)'], 'us-west-2')
df

df.AvailabilityZone.value_counts()

df.ProductDescription.value_counts()

df.dtypes

## Select a particular criteria for analysis

uwta = df.loc[df['AvailabilityZone'] == 'us-west-2a']
uwta

uwta_m4 = uwta.loc[uwta['InstanceType'] == 'm4.large']
uwta_m4

## Setting the timestamp as index for plotting price trends
uwta_m4.set_index('Timestamp',inplace=True)

## dropping other columns 
for col in ['InstanceType', 'AvailabilityZone', 'ProductDescription']:
    uwta_m4 = uwta_m4.drop(col, axis=1)

## Converting the dtype of spot price to numberic
uwta_m4['SpotPrice'] = uwta_m4['SpotPrice'].apply(pd.to_numeric)

## resample the data with the required frequency
uwta_m4_day = uwta_m4.resample('D').mean()

## resampling with frequency hour
uwta_m4_hour = uwta_m4.resample('H').mean()

uwta_m4_hour

uwta_m4_hour.fillna(method='ffill',inplace=True)
uwta_m4_hour

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

## plotting for the day frequency; remember we did not fill the values for day frequency
plt.plot(uwta_m4_day)

plt.plot(uwta_m4_hour)

## Rolling statistics
rolmean = uwta_m4_day.rolling(window = 7).mean()
rolstd = uwta_m4_day.rolling(window = 7).std()

## plot the results
orig = plt.plot(uwta_m4_day, color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)

## Dickney Fuller test for stationarity
from statsmodels.tsa.stattools import adfuller

x = uwta_m4_day.SpotPrice
x

res = adfuller(x)

dfoutput = pd.Series(res[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in res[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)



