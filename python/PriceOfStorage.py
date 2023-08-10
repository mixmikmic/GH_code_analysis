import pandas as pd
data = '../data/memory-price.csv'
pd.read_csv(data).head()

from altair import *

DriveSize = Formula(field='DriveSize',
                    expr=('{x}<1E3?"MB":{x}<1E6?"GB":"TB"'
                          ''.format(x='datum.sizeInMb')))

Chart(data).mark_circle().encode(
    x=X('date:T', timeUnit='year', axis=Axis(title=' ')),
    y=Y('dollarsPerGb:Q', scale=Scale(type='log'),
        axis=Axis(grid=False, format='$', title='Cost per GB')),
    color=Color('DriveSize:N',
                scale=Scale(domain=['MB', 'GB', 'TB']))
).transform_data(
    calculate=[DriveSize]
)

