import pandas as pd
import statsmodels.formula.api as sm
import statsmodels.stats.sandwich_covariance as sw
import numpy as np

from urllib import urlopen

filehandle = urlopen('http://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.txt')
df = pd.read_table(filehandle, names=['firmid','year','x','y'],
                   delim_whitespace=True)
df

ols = sm.ols(formula='y ~ x', data=df).fit(use_t=True)
ols.summary()

robust_ols = sm.ols(formula='y ~ x', data=df).fit(cov_type='HC1', use_t=True)
robust_ols.summary()

cluster_firm_ols = sm.ols(formula='y ~ x', data=df).fit(cov_type='cluster',
                                                        cov_kwds={'groups': df['firmid']},
                                                        use_t=True)
cluster_firm_ols.summary()

cluster_year_ols = sm.ols(formula='y ~ x', data=df).fit(cov_type='cluster',
                                                        cov_kwds={'groups': df['year']},
                                                        use_t=True)
cluster_year_ols.summary()

cluster_2ways_ols = sm.ols(formula='y ~ x', data=df).fit(cov_type='cluster',
                                                         cov_kwds={'groups': np.array(df[['firmid', 'year']])},
                                                         use_t=True)
cluster_2ways_ols.summary()

from rpy2.robjects import pandas2ri
import rpy2.robjects as ro

# Hide warnings (R is quite verbose). Comment out to keep the warnings
ro.r['options'](warn=-1)

pandas2ri.activate()

# model is a string (R-style regression model)
# clusters is a list a strings
# returns a pandas DataFrame
def multiway_cluster(df, model, clusters):
    rdf = pandas2ri.py2ri(df)
    ro.globalenv['rdf'] = rdf

    clusters_grp = ' + '.join(['rdf$' + x for x in clusters])
    reg_command = 'reg <- lm(' + model + ', data = rdf)\n'
    vcov_command = 'reg$vcov <- cluster.vcov(reg, ~ ' + clusters_grp + ')\n'
    libraries = '''
library(zoo)
library(multiwayvcov)
library(lmtest)
'''
    output = '''
result <- coeftest(reg, reg$vcov)
regnames <- attributes(result)$dimnames[[1]]
colnames <- attributes(result)$dimnames[[2]]
'''

    command = libraries + reg_command + vcov_command + output
    ro.r(command)

    res = pd.DataFrame(ro.r('result'))
    res.columns = ro.r('colnames')
    res.index = ro.r('regnames')
    
    return res

multiway_cluster(df, 'y ~ x', clusters=['firmid', 'year'])

# Note: R in quite verbose with the warnings, but most of them aren't really warnings.

firm_fe_ols = sm.ols(formula='y ~ x + C(firmid)', data=df).fit(use_t=True)
#firm_fe_ols.summary()  
# The summary is ommitted because the large number 
# of dummy variables make it unpleasant to look at.

year_fe_ols = sm.ols(formula='y ~ x + C(year)', data=df).fit(use_t=True)
year_fe_ols.summary()

firm_year_fe_ols = sm.ols(formula='y ~ x + C(firmid) + C(year)', data=df).fit(use_t=True)
#firm_year_fe_ols.summary() 
# The summary is ommitted because the large number 
# of dummy variables make it unpleasant to look at.

firm_cluster_year_fe_ols = sm.ols(formula='y ~ x + C(year)', data=df).fit(cov_type='cluster',
                                                                          cov_kwds={'groups': df['firmid']},
                                                                          use_t=True)
firm_cluster_year_fe_ols.summary()

def fama_macbeth(formula, time_label, df, lags=3):
    res = df.groupby(time_label).apply(lambda x: sm.ols(formula=formula,
                                                     data=x).fit())

    l = [x.params for x in res]
    p = pd.DataFrame(l)

    means = {}
    params_labels = res.iloc[0].params.index

    # The ':' character used by patsy doesn't play well with pandas column names.
    p.columns = [x.replace(':', '_INTER_') for x in p.columns]

    for x in p.columns:
        if lags is 0:
            means[x.replace('_INTER_',':')] = sm.ols(formula=x + ' ~ 1',
                                                     data=p[[x]]).fit(use_t=True)
        else:
            means[x.replace('_INTER_',':')] = sm.ols(formula=x + ' ~ 1',
                                                     data=p[[x]]).fit(cov_type='HAC',
                                                                      cov_kwds={'maxlags': lags},
                                                                      use_t=True)

    params = []
    stderrs = []
    tvalues = []
    pvalues = []
    for x in params_labels:
        params.append(means[x].params['Intercept'])
        stderrs.append(means[x].bse['Intercept'])
        tvalues.append(means[x].tvalues['Intercept'])
        pvalues.append(means[x].pvalues['Intercept'])

    result = pd.DataFrame([params, stderrs, tvalues, pvalues]).T
    result.index = params_labels
    result.columns = ['coef', 'stderr', 'tvalue', 'pvalue']
    result['stars'] = ''
    result.loc[result.pvalue < 0.1, 'stars'] = '*'
    result.loc[result.pvalue < 0.05, 'stars'] = '**'
    result.loc[result.pvalue < 0.01, 'stars'] = '***'

    return result

fama_macbeth('y ~ x', 'year', df)

fama_macbeth('y ~ x', 'year', df, lags=0)

# Note: this adjustment doesn't really make sense for our sample dataset, it's just an illustration.
nw_ols = sm.ols(formula='y ~ x', data=df).fit(cov_type='HAC',
                                              cov_kwds={'maxlags': 3},
                                              use_t=True)
nw_ols.summary()

dk_ols = sm.ols(formula='y ~ x', data=df).fit(cov_type='nw-groupsum',
                                              cov_kwds={'time': df.year,
                                                        'groups': df.firmid,
                                                        'maxlags': 5},
                                              use_t=True)
dk_ols.summary()



