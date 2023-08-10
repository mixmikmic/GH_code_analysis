from __future__ import print_function, division

get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np

import thinkstats2
import thinkplot
import utils

import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('whitegrid')

gss = utils.ReadGss('gss_religion_data')
print(gss.shape)
gss.head()

sample = utils.ResampleByYear(gss)
sample.head()

utils.fill_missing(sample, 'educ')

utils.values(sample, 'educ')

sample['college'] = sample.educ >= 13
sample.college.mean()

utils.fill_missing(sample, 'relig')

utils.values(sample, 'relig')

sample['none'] = sample.relig ==4
sample.none.mean()

grouped = sample.groupby(['year', 'college'])

percent_none = grouped.none.mean().unstack()
percent_none.plot()
plt.ylabel('Fraction with no affiliation')
plt.xlim([1970, 2018]);

diff = percent_none[True] - percent_none[False]
plt.plot(diff)
plt.xlabel('Year');
plt.ylabel('Difference in fraction with no affiliation')
plt.xlim([1970, 2018]);

college_none = 30.9
no_college_none = college_none - 6
fraction_college = 0.65
fraction_no_college = 1-fraction_college

fraction_none = fraction_college * college_none + fraction_no_college * no_college_none
fraction_none

college_none - 0.35 * 6





