import pandas as pd
import numpy as np

from plotnine import *
from plotnine.data import economics

from mizani.breaks import date_breaks
from mizani.formatters import date_format

theme_set(theme_linedraw()) # default theme

get_ipython().magic('matplotlib inline')

economics.head()

(ggplot(economics)
 + geom_point(aes('date', 'psavert'))
 + labs(y='personal saving rate')
)

(ggplot(economics)
 + geom_point(aes('date', 'psavert'))
 + scale_x_datetime(breaks=date_breaks('10 years'))        # new
 + labs(y='personal saving rate')
)


(ggplot(economics)
 + geom_point(aes('date', 'psavert'))
 + scale_x_datetime(breaks=date_breaks('10 years'), labels=date_format('%Y'))     # modified
 + labs(y='personal saving rate')
)

def custom_date_format1(breaks):
    """
    Function to format the date
    """
    return [x.year if x.month==1 and x.day==1 else "" for x in breaks]

(ggplot(economics)
 + geom_point(aes('date', 'psavert'))
 + scale_x_datetime(                                # modified
     breaks=date_breaks('10 years'),
     labels=custom_date_format1)
 + labs(y='personal saving rate')
)

from datetime import date

def custom_date_format2(breaks):
    """
    Function to format the date
    """
    res = []
    for x in breaks:
        # First day of the year
        if x.month == 1 and x.day == 1:
            fmt = '%Y'
        # Every other month
        elif x.month % 2 != 0:
            fmt = '%b'
        else:
            fmt = ''
            
        res.append(date.strftime(x, fmt))
            
    return res

(ggplot(economics.loc[40:60, :])                            # modified
 + geom_point(aes('date', 'psavert'))
 + scale_x_datetime(
     breaks=date_breaks('1 months'),
     labels=custom_date_format2,
     minor_breaks=[])
 + labs(y='personal saving rate')
)

def custom_date_format3(breaks):
    """
    Function to format the date
    """
    res = []
    for x in breaks:
        # First day of the year
        if x.month == 1:
            fmt = '%Y'
        else:
            fmt = '%b'
            
        res.append(date.strftime(x, fmt))
            
    return res


def custom_date_breaks(width=None):
    """
    Create a function that calculates date breaks
    
    It delegates the work to `date_breaks`
    """
    def filter_func(limits):
        breaks = date_breaks(width)(limits)
        # filter
        return [x for x in breaks if x.month % 2]
    
    return filter_func


(ggplot(economics.loc[40:60, :])
 + geom_point(aes('date', 'psavert'))
 + scale_x_datetime(                                        # modified
     breaks=custom_date_breaks('1 months'),
     labels=custom_date_format3)
 + labs(y='personal saving rate')
)

