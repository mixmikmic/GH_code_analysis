import pandas as pd
import numpy as np

from plotnine import *
from plotnine.data import *

get_ipython().magic('matplotlib inline')

mpg.head()

p = ggplot(aes(x='displ', y='cty'), mpg)
p + geom_point()

p + geom_point(aes(color='factor(cyl)'))

p + geom_point(aes(shape='factor(cyl)'))

p + geom_point(aes(color='hwy'))

p + geom_point(aes(color='hwy')) + scale_color_gradient(low='blue', high='red')

p + geom_point(aes(size='hwy'))

