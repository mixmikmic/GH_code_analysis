from datetime import datetime
from datetime import timedelta
import pandas as pd

# A:
now = datetime.now()
print(now)

datetime_df = pd.DataFrame([now], columns=['Time'])

datetime_df.dtypes

datetime_df.Time.dt.weekday_name.head()

# A:
def day_of_week(times):
    datetime_df = pd.DataFrame(times)

# A:

# A:

import pandas as pd
from datetime import timedelta
get_ipython().magic('matplotlib inline')

# A:

# A:

# A:

# A:

# A:

# A:

# A:

index = pd.date_range('3/1/2016', '6/1/2016')

# Specifify a start point and how many periods after
pd.date_range(start='3/1/2016', periods=20)

#Specify a end point and how many periods before
pd.date_range(end='6/1/2016', periods=20)

# Frequency specifyins the length of the periods the default 'D' being daily.  I imagine BM is Bi-Monthly
pd.date_range('1/1/2016', '12/1/2016', freq='BM')

pd.date_range('3/7/2016 12:56:31', periods=6)
# normalize creates normal daily times, and will make the default time for each day midnight.
pd.date_range('3/7/2012 12:56:31', periods=6, normalize=True)

# March 2016 was our start period, and the period frequency is months.
march_2016 = pd.Period('2016-03', freq='M')

print march_2016.start_time
print march_2016.end_time

