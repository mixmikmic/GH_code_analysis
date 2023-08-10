import numpy as np
import pandas as pd
import krisk.plot as kk
from krisk import init_notebook; init_notebook()

np.random.seed(0)
df = pd.DataFrame({'val': -1 + 10 * np.random.randn(10)})

df.head()

kk.waterfall(df['val'])

kk.waterfall(df['val'], color_coded=True, annotate="outside", up_name="up")

url = 'http://www.stat.ubc.ca/~jenny/notOcto/STAT545A/examples/gapminder/data/gapminderDataFiveYear.txt'
gapminder = (pd.read_csv(url,sep='\t')
             .groupby(['year','continent'], as_index=False)
             .first())

gapminder.head()

kk.bar_line(gapminder, 'continent', 'lifeExp', 'gdpPercap')

kk.bar_line(gapminder, 'continent', 'lifeExp', 'gdpPercap', is_distinct=True)

