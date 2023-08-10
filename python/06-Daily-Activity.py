import pandas as pd
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

data = [['2016-07-01',19561],
        ['2016-07-02',14468],
        ['2016-07-03',12204],
        ['2016-07-04',17766],
        ['2016-07-05',22821],
        ['2016-07-06',27024],
        ['2016-07-07',25088],
        ['2016-07-08',23526],
        ['2016-07-09',13720],
        ['2016-07-10',15210],
        ['2016-07-11',24669],
        ['2016-07-12',24946],
        ['2016-07-13',28998],
        ['2016-07-14',27755],
        ['2016-07-15',22488],
        ['2016-07-16',15160],
        ['2016-07-17',12991],
        ['2016-07-18',22446],
        ['2016-07-19',28449],
        ['2016-07-20',26296],
        ['2016-07-21',26433],
        ['2016-07-22',27042],
        ['2016-07-23',14400],
        ['2016-07-24',12409],
        ['2016-07-25',19668]]

df = pd.DataFrame(data, columns=['Day', 'Tweets'])

df['Day'] = pd.to_datetime(df['Day']).dt.weekday

df = df.groupby(df['Day']).mean().round()

df['Day'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

df

x = range(len(df['Day']))
h = plt.bar(x, df['Tweets'])
xticks_pos = [0.5*patch.get_width() + patch.get_xy()[0] for patch in h]
plt.xticks(xticks_pos, df['Day'])
plt.show()

data = [['2016-07-01',1393],
        ['2016-07-02',1000],
        ['2016-07-03',1000],
        ['2016-07-04',1538],
        ['2016-07-05',1600],
        ['2016-07-06',1928],
        ['2016-07-07',1867],
        ['2016-07-08',1710],
        ['2016-07-09',982],
        ['2016-07-10',1099],
        ['2016-07-11',1718],
        ['2016-07-12',1781],
        ['2016-07-13',1799],
        ['2016-07-14',1774],
        ['2016-07-15',1535],
        ['2016-07-16',1118],
        ['2016-07-17',1063],
        ['2016-07-18',1809],
        ['2016-07-19',1864],
        ['2016-07-20',1938],
        ['2016-07-21',1877],
        ['2016-07-22',1723],
        ['2016-07-23',1013],
        ['2016-07-24',1010],
        ['2016-07-25',1563]]

df = pd.DataFrame(data, columns=['Day', 'Tweets'])

df['Day'] = pd.to_datetime(df['Day']).dt.weekday

df = df.groupby(df['Day']).mean().round()

df['Day'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

df

x = range(len(df['Day']))
h = plt.bar(x, df['Tweets'])
xticks_pos = [0.5*patch.get_width() + patch.get_xy()[0] for patch in h]
plt.xticks(xticks_pos, df['Day'])
plt.show()



