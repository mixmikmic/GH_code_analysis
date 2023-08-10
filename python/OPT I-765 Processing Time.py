get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")

import csv
import pandas

data_eac = pandas.read_csv('raw_data_eac.csv')
data_ysc = pandas.read_csv('raw_data_ysc.csv')
data = data_eac.append(data_ysc)

data

data.groupby('title').count()['receipt']

def gen_status(row):
    if 'I-765' in row['text'] and 'Case Was Received' in row['title']:
        return 'processing'
    if 'I-765' in row['text'] and 'Case Was Transferred' in row['title']:
        return 'processing'
    if 'I-765' in row['text'] and 'Case Was Rejected' in row['title']:
        return 'rejected'
    if 'I-765' in row['text'] and 'Case Was Approved' in row['title']:
        return 'approved'
    if row['title'] == 'Card Was Delivered To Me By The Post Office':
        return 'approved'
    if row['title'] == 'New Card Is Being Produced':
        return 'approved'
    return ''

data_24['status'] = data_24.apply(gen_status, axis=1)
data_26['status'] = data_26.apply(gen_status, axis=1)

import re
import numpy
pattern = re.compile('On ([A-z]+ [0-9]+, [0-9]+),')

last_date = None
def extract_date(row):
    if row['status'] == 'processing':
        m = pattern.search(row['text'])
        global last_date
        last_date = numpy.datetime64(pandas.Timestamp(m.group(1)))
        return last_date
    return last_date

data['date'] = data.apply(extract_date, axis=1)

data

result = pandas.DataFrame(data[data['status'] == 'processing'].groupby('date')['receipt'].count())
result['approved'] = data[data['status'] == 'approved'].groupby('date')['receipt'].count()
result['rejected'] = data[data['status'] == 'rejected'].groupby('date')['receipt'].count()
result.columns = ['Processing', 'Approved', 'Rejected']

result.plot()

result.loc['2016-03-30']

result['Total'] = result['Processing'] + result['Approved'] + result['Rejected']
result['Approved Proportion'] = result['Approved'] / result['Total']
result['Approved Proportion'].plot()

