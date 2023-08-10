import pandas as pd
import numpy as np

data = '../data/births.csv'
pd.read_csv(data).head()

from altair import *

genderscale = Scale(domain=['M', 'F'], range=["#659CCA", "#EA98D2"])

Chart(data).mark_line().encode(
    X('date:T', timeUnit='year'),
    Y('mean(births):Q'),
    Color('gender:N', scale=genderscale)
)

Chart(data).mark_line().encode(
    X('date:T', timeUnit='day', axis=Axis(title=' ')),
    Y('mean(births):Q', scale=Scale(zero=False)),
    Color('gender:N', scale=genderscale)
).configure_scale(bandSize=80)

Chart(data).mark_line().encode(
    X('date:T', timeUnit='date',
      axis=Axis(title=' ', grid=False, labels=False,
                axisColor='white', tickColor='white')),
    Y('mean(births):Q', scale=Scale(zero=False)),
    Column('date:T', timeUnit='month', scale=Scale(padding=0)),
    Color('gender:N', scale=genderscale)
).configure_cell(
    width=55
).configure_facet_cell(
    strokeOpacity=0.3
)

