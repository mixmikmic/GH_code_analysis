# Visit the following link to download the dataset in your local machine: 
# https://raw.githubusercontent.com/blast-analytics-marketing/RFM-analysis/master/sample-orders.csv

import pandas as pd
orders = pd.read_csv('sample-data.csv',sep=',')

# Exploring the dataset
orders.head()

# You have a Text object. The strftime function requires a datetime object.
# The code below takes an intermediate step of converting your Text to a datetime using strptime.

import datetime
testeddate = '2014/12/31'
NOW = datetime.datetime.strptime(testeddate,'%Y/%m/%d')

# Convert the date_placed column into datetime

orders['order_date'] = pd.to_datetime(orders['order_date'])

rfmTable = orders.groupby('customer').agg({'order_date': lambda x: (NOW - x.max()), # Recency
                                        'order_id': lambda x: len(x),               # Frequency
                                        'grand_total': lambda x: x.sum()})          # Monetary Value

rfmTable.rename(columns={'order_date': 'recency', 
                         'order_id': 'frequency', 
                         'grand_total': 'monetary_value'}, inplace=True)


# Converting the time delta to days instead of including the term 'days' in the actual column.

rfmTable['recency'] = rfmTable['recency'].astype('timedelta64[D]')

# Checking the results

rfmTable.head()

aaron = orders[orders['customer']=='Aaron Bergman']
aaron

(NOW - datetime.datetime(2013,11,11)).days==415

quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])
quantiles

quantiles = quantiles.to_dict()
quantiles

rfmSegmentation = rfmTable

# Arguments (x = value, p = recency, d = monetary_value, frequency, k = quartiles dict)

def RClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
# --------------------------------------------------------------------------------------------------------------------------#    


# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)

def FMClass(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1

rfmSegmentation['R_Quartile'] = rfmSegmentation['recency'].apply(RClass, args=('recency',quantiles,))
rfmSegmentation['F_Quartile'] = rfmSegmentation['frequency'].apply(FMClass, args=('frequency',quantiles,))
rfmSegmentation['M_Quartile'] = rfmSegmentation['monetary_value'].apply(FMClass, args=('monetary_value',quantiles,))

rfmSegmentation['RFMClass'] = rfmSegmentation.R_Quartile.map(str)                             + rfmSegmentation.F_Quartile.map(str)                             + rfmSegmentation.M_Quartile.map(str)

rfmSegmentation.head()

# rfmSegmentation.to_clipboard()
#rfmSegmentation.to_csv('rfm-table.csv', sep=',')

rfmSegmentation[rfmSegmentation['RFMClass']=='111'].sort('monetary_value', ascending=False).head(5)

rfmSegmentation['Total Score'] = rfmSegmentation['R_Quartile'] + rfmSegmentation['F_Quartile'] +rfmSegmentation['M_Quartile']

rfmSegmentation.head()

# Setting up the label for each client and adding the column "Label" to the dataframe

label = [0] * len(rfmSegmentation)

for i in range(0,len(rfmSegmentation)):

    if rfmSegmentation['Total Score'][i] == 12:
        label[i] = "Excellent"
        
    elif rfmSegmentation['Total Score'][i] >= 7 :
        label[i] = "Good"
        
    elif rfmSegmentation['Total Score'][i] >= 3:
        label[i] = "Bad"
        
    else:
        label[i] = "Only 1 transaction?"        

# Adding the 'Label' column to our dataframe

rfmSegmentation['Label'] = label

# Count the frequency that a value occurs in a dataframe column for the labels.

rfmSegmentation['Label'].value_counts()


def color(val):
    if val == "Excellent":
        color = 'green'
    elif val == "Good":
        color = 'yellow'
    elif val == "Bad":
        color = 'red'
    return 'background-color: %s' % color

rfmSegmentation.style.applymap(color, subset=['Label'])

rfmSegmentation.to_csv('rfm-table-data.csv', sep=',')

