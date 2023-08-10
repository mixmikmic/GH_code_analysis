get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-whitegrid')
mpl.rcParams['figure.figsize'] = (12.5, 6.0)
import pandas as pd
pd.options.display.max_rows = 12

country = ['Northern Ireland', 'Scotland', 'Wales', 'England', 'Isle of Man']
capital = ['Belfast', 'Edinburgh', 'Cardiff', 'London', 'Douglas']
area = [14130, 77933, 20779, 130279, 572]
population2017 = [1876695, 5404700, np.nan, 55268100, np.nan]
population2011 = [1810863, 5313600, 3063456, 53012456, 83314]
uk_area = pd.Series(area, index=country)
uk_area.sort_index(inplace=True)
array = np.array([area, capital, population2011, population2017]).T
data = pd.DataFrame({'capital': capital,
                     'area': area,
                     'population 2011': population2011,
                     'population 2017': population2017},
                    index=country)
data.sort_index(inplace=True)
data

pop = data[['population 2011', 'population 2017']]
pop.columns = [2011, 2017]
pop

pop_year = pop.stack()
pop_year

pop_year['England']

pop_year[('England', 2011)]

pop_year[:, 2017]

pop_year.unstack()

pop.index.name = 'country'
pop_full = pop.reset_index()
pop_full

pop_melt = pop_full.melt(id_vars=['country'], var_name='year')
pop_melt

pop_full = pop_melt.pivot(index='country', columns='year', values='value')
pop_full

pop_agg = pop_melt.pivot_table(index='country', aggfunc=[np.mean, lambda x: np.sum(~np.isnan(x))])
pop_agg

pop_agg.columns = ['mean', 'not null']
pop_agg

