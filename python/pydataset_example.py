get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from pydataset import data
import pandas as pd
pd.set_option("display.max_rows",1000)
sns.set(style="ticks")

df = data('cars')

df.head()

sns.boxplot(data=df, palette='PRGn')
sns.despine(offset=10, trim=True)

sns.palplot(sns.color_palette("PRGn"))

sns.palplot(sns.light_palette('blue'))

df.boxplot(column='Reaction', by='Days')

choose_cubehelix_palette('s')

