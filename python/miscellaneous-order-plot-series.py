import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from plotnine import *
from plotnine.data import mpg

get_ipython().magic('matplotlib inline')

(ggplot(mpg)
 + aes(x='manufacturer') 
 + geom_bar(size=20)
 + coord_flip()
 + labs(y='Count', x='Manufacturer', title='Number of Cars by Make')
)

# Determine order and create a categorical type
manufacturer_list = mpg['manufacturer'].value_counts().index.tolist()
manufacturer_cat = CategoricalDtype(categories=manufacturer_list, ordered=True)

# Cast the existing categories into the new category. Due to a bug in pandas
# we need to do this via a string. 
mpg['manufacturer_cat'] = mpg['manufacturer'].astype(str).astype(manufacturer_cat)

(ggplot(mpg)
 + aes(x='manufacturer_cat')
 + geom_bar(size=20)
 + coord_flip()
 + labs(y='Count', x='Manufacturer', title='Number of Cars by Make')
)

