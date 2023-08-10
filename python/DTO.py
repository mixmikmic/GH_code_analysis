# import the packages
import pandas as pd
# First thing is to load the dataset
df = pd.read_csv('data.csv', low_memory = False)
df.drop('Unnamed: 0', axis=1, inplace=True)
df.head(4)

# we will be looking into DateAdded, and Last Open. If LastOpen is NaN, this means that email was not open at all
# we need to drop the dates, and leave only the time
# Now, in my opinion to make it easier, we should adjust the DateAdded to the range from 10:00 to 10:59
df['hour'] = df['hour'].str[10:]
df['hour'] = df['hour'].str[:3]
df.head(4)

# Excellent
# now, last open can be changed into Boolean, since we don't really need to know when it was opened
# we just need to know wether they were opened/clicked or not.
df['Opened'] = pd.notnull(df['LastOpen'])
df.head(4)

# the hours column can be optimized by making it catagorical
# also, you may want to correct hour extraction; df.hour[0]
df.hour = df.hour.astype('category')

# here is one way to build a mapping of hour to open ration
hour_ratio_lookup = {}
for hour in df.hour.unique():
    sdf = df[df.hour == hour]
    n_total_emails = len(sdf)
    n_opened_emails = sdf.Opened.sum()
    hour_ratio_lookup[hour] = n_opened_emails / n_total_emails

# printing the dictionary
print(hour_ratio_lookup)
ser = pd.Series(hour_ratio_lookup)
ndf = pd.DataFrame(columns=['hour', 'open_ratio'])
ndf.hour, ndf.open_ratio = ser.index, ser.values
# now let's apply the mapping and add the ratio column
df['open_ratio'] = df.hour.map(hour_ratio_lookup)

ndf.head()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

ndf.plot.bar(x='hour', y='open_ratio')



