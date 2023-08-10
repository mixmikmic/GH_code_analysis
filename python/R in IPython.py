import sys
sys.executable

get_ipython().magic('load_ext rpy2.ipython')

get_ipython().run_cell_magic('R', '-o mtcars', 'library(ggplot2)\nmtcars')

mtcars

type(mtcars)

mtcars.plot(x='hp', y='mpg', kind='scatter', marker='o')

import numpy as np
import pandas as pd

data = [ np.cumsum(np.random.random(1000) - 0.49) for i in range(3) ]
df = pd.DataFrame.from_records(data, index=list('ABC')).T
df.plot()

df

get_ipython().run_cell_magic('R', '-i df', 'summary(df)')

get_ipython().run_cell_magic('R', '', 'qplot(seq_along(df$A), df$A) + geom_line()')

