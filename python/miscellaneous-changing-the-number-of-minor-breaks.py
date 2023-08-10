import pandas as pd
import numpy as np
from plotnine import *
from plotnine.data import mtcars
from mizani.breaks import minor_breaks
from mizani.transforms import identity_trans

get_ipython().magic('matplotlib inline')

original_theme = theme_set(
    theme_gray()
    + theme(panel_grid_minor_y=element_line(color='brown', linetype='dashed', alpha=.5))
)

(ggplot(mtcars)
 + geom_point(aes('wt', 'mpg'))
)

(ggplot(mtcars)
 + geom_point(aes('wt', 'mpg'))
 + scale_y_continuous(minor_breaks=minor_breaks(4))
)

(ggplot(mtcars)
 + geom_point(aes('wt', 'mpg'))
 + scale_y_continuous(trans=identity_trans(minor_breaks=minor_breaks(4)))
)

