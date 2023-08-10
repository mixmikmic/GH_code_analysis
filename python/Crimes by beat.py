from datetime import datetime, timedelta
pivot = datetime.strptime('11/18/2014', '%m/%d/%Y')
today = datetime.strptime('1/18/2016', '%m/%d/%Y')
print today - pivot

period = timedelta(days=426)
print pivot - period

import pandas as pd
url = 'https://data.cityofchicago.org/api/views/qa42-2iy9/rows.csv?accessType=DOWNLOAD'
frame = pd.read_csv(url, parse_dates=['Date'])
print frame.head(2)
print '%d crimes found' % len(frame)

frame['Date Only'] = pd.to_datetime(frame['Date'].apply(lambda x: x.date()))
pivot = pivot.date()

print '%d crimes on or after %s' % (frame[frame['Date Only'] >= pivot].Date.count(), pivot)

print '%d crimes before %s' % (frame[frame['Date Only'] < pivot].Date.count(), pivot)

# Let's get nicer-looking plots. Can't use ggplot because my version of matplotlib is too old (I think).
pd.set_option('display.mpl_style', 'default') 
pd.set_option('display.width', 10000) 
pd.set_option('display.max_columns', 60)
# We need to specifically ask matplotlib to display plots inline
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
frame.groupby('Date Only').count().plot(legend=None)

for ucr in frame['IUCR'].unique():
    print ucr
    ucr_frame = frame[frame['IUCR'] == ucr]
    print '%d crimes on or after %s' % (ucr_frame[ucr_frame['Date Only'] >= pivot].Date.count(), pivot)
    print '%d crimes before %s' % (ucr_frame[ucr_frame['Date Only'] < pivot].Date.count(), pivot)
    print '---'

