import matplotlib as mpl
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
mpl.style.use('bmh')

import numpy as np
import statsmodels.api as sm

def lin_reg(y):
    X = np.linspace(0, 1, len(y))
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit()

import numpy as np

from scipy.stats import linregress

def plot_histogram(text, bins=50, w=4, h=2):
    
    author = '{} {}'.format(text.get('authorFirst'), text.get('authorLast'))
    print('{} ({}, {})'.format(text['title'], author, text['year']))
    
    plt.figure(figsize=(w,h))
    
    y, _, _ = plt.hist(text['offsets'], bins, (0,1))
    
    fit = lin_reg(y)
    
    slope = fit.params[1]
    p = fit.pvalues[1]
    print(slope, p)
    
    x1 = 0
    x2 = 1
    y1 = fit.predict()[0]
    y2 = fit.predict()[-1]
    
    plt.plot([x1, x2], [y1, y2])
    plt.show()

from lint_analysis.token_offsets import Dataset

ds = Dataset.from_local('a.json')

texts = ds.texts()

from scipy.stats import histogram

data = []
for text in ds.texts():
    
    y, _, _, _ = histogram(text['offsets'], 50, (0, 1))
    fit = lin_reg(y)
    
    slope = fit.params[1]
    p = fit.pvalues[1]
    
    data.append((slope, p, text))

import pandas as pd

df = pd.DataFrame(data, columns=('slope', 'p', 'text'))

p05 = df[df.p < 0.05]
p01 = df[df.p < 0.01]

len(df), len(p05), len(p01)

len(p01[p01.slope > 0]) / len(p01[p01.slope < 0])

down = p01.sort_values('slope')
up = p01.sort_values('slope', ascending=False)

for text in down['text'][:50]:
    plot_histogram(text)

for text in up['text'][:50]:
    plot_histogram(text, w=8, h=4)



