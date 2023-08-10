import pandas as pd
import numpy as np
import pylab as plt
import os
import datetime
get_ipython().magic('matplotlib inline')
os.chdir('D:\Python\Catalina Challenge')
df = pd.read_csv("data_science_challenge_samp_18.csv")
df.dtypes

df.head()

df['order_date'] = pd.to_datetime(df['order_date'])
df.dtypes

#Applying per column:
def num_missing(x):
  return sum(x.isnull())
print("Missing values per column:")
print(df.apply(num_missing, axis=0)) #axis=0 defines that function is to be applied on each column

df['cust_id'].unique().size

# Group data by Customer ID
a = df.groupby(['cust_id']).sum()
a = a.reset_index()
plt.xlabel('Customers', fontsize=18)
plt.ylabel('Total Spend', fontsize=16)
plt.scatter(a['cust_id'],a['total_spend'],color='k',alpha=.2,s=2)

plt.xlabel('Customers', fontsize=18)
plt.ylabel('Total Units Purchased', fontsize=16)
plt.scatter(a['cust_id'],a['units_purchased'],color='k',alpha=.2,s=2)

df['order_date'] = pd.to_datetime(df['order_date'])

sales_by_day = df.groupby(['order_date']).sum()
sales_by_day = sales_by_day.reset_index()
sales_by_day.dtypes

x = sales_by_day['order_date']
y = sales_by_day['total_spend']
plt.plot_date(x, y, xdate='True')
plt.xlabel('Day', fontsize=18)
plt.ylabel('Total Spend', fontsize=16)
#plt.plot_date(x, y, xdate='True')

df2 = df
# Assign a week number to each date
df2['weekNo'] = df2['order_date'].dt.week
df2['dayOfWeek'] = df2['order_date'].dt.weekday_name
#Check sales by week
weekly_sales = df2.groupby('weekNo').sum()
weekly_sales = weekly_sales.reset_index()
weekly_units = pd.Series(weekly_sales['units_purchased'], index=weekly_sales['weekNo'])
weekly_units.plot.bar()

#Check sales by day of week
sale_by_day = df2.groupby('dayOfWeek').sum()
sale_by_day = sale_by_day.reset_index() 

sale_by_day
DayOfWeekOfCall = [0,1,2,3,4,5,6]
LABELS = ["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]
plt.bar(DayOfWeekOfCall, sale_by_day['units_purchased'], align='center')
plt.xticks(DayOfWeekOfCall, LABELS)
plt.xlabel('Day of Week', fontsize=18)
plt.ylabel('Units purchased', fontsize=16)
plt.show()
#x = sale_by_day['dayOfWeek']
#y = sale_by_day['units_purchased']
#plt.plot_date(x, y, xdate='True')

from IPython.core.display import Image
Image(filename=('C:/Users/Hitanshu/Pictures/RFM.jpg'))

from lifetimes.utils import summary_data_from_transaction_data

summary = summary_data_from_transaction_data(df, 'cust_id', 'order_date','units_purchased', observation_period_end='2016-03-27')
summary.head()

from lifetimes import BetaGeoFitter

# similar API to scikit-learn and lifelines.
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])

from lifetimes.plotting import plot_frequency_recency_matrix

plot_frequency_recency_matrix(bgf)

from lifetimes.plotting import plot_probability_alive_matrix

plot_probability_alive_matrix(bgf)

#Ranking customers from best to worst
#Let's return to our customers and rank them from "highest expected purchases in the next period" to lowest. Models expose a method that will predict a customer's expected purchases in the next period using their history.
t = 1
summary['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, summary['frequency'], summary['recency'], summary['T'])
summary.sort_values(by='predicted_purchases').tail(5)

from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf)

from lifetimes.utils import calibration_and_holdout_data

summary_cal_holdout = calibration_and_holdout_data(df, 'cust_id', 'order_date',
                                        calibration_period_end='2015-12-31',
                                        observation_period_end='2016-03-27' )   
summary_cal_holdout.head()

from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases

bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])
plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout)

t = 7 #predict purchases in 7 periods (in this case days)
individual = summary.iloc[10000]
# The below function is an alias to `bfg.conditional_expected_number_of_purchases_up_to_time`
bgf.predict(t, individual['frequency'], individual['recency'], individual['T'])

summary_with_units_value = summary_data_from_transaction_data(df, 'cust_id', 'order_date','units_purchased', observation_period_end='2016-03-27')
summary_with_units_value.head()

#Filter by Purchase frequency > 0
returning_customers_summary = summary_with_units_value[summary_with_units_value['frequency']>0]

#Filter by Units Purchased < 3
returning_customers_summary2 = returning_customers_summary[returning_customers_summary['monetary_value']<3]
returning_customers_summary2.head()

bgf.predict(t, returning_customers_summary2['frequency'], returning_customers_summary2['recency'], returning_customers_summary2['T'])
# t = 7 days

