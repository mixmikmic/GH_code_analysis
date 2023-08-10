from pathlib import Path
import pandas as pd
import numpy as np
import random 

import statsmodels.formula.api as sm
from statsmodels.graphics.gofplots import ProbPlot
import math
from sklearn import linear_model

from matplotlib import pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)

import datetime
import sys

### for debugging purposes
###sys.version
### should say:
###'3.6.3 |Anaconda custom (64-bit)| (default, Oct  6 2017, 12:04:38) \n
###[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)]'


### begin ETL

d = pd.read_csv('../data/raw_data.csv')
d_working = d.dropna(how='any')


### create numeric mapping for locations
unique_locs = d.Location.value_counts().index.tolist()
loc_map = [x for x in range(len(unique_locs))]
loc_set = dict(zip(unique_locs, loc_map))
#loc_set

d_working['numeric_location'] = d_working['Location'].map(lambda x: loc_set[x] if x in loc_set.keys() else -1)

### get dummies for locations
dum = pd.get_dummies(d_working['numeric_location'])
dum.columns = ['loc_%s' % (x) for x in dum.columns.tolist()]
d_working = pd.concat([d_working, dum], axis = 1)

### create week aggregates

def week_num(date_string):
    date = datetime.datetime.strptime(date_string,'%m/%d/%y')
    return datetime.date.isocalendar(date)[1]

d_working['week_number'] = d_working['Date'].apply(week_num)


### create a calculated field that staggers the google trends to the next day
d_working['stagger_second_flu_trend'] = [0] + d_working['google_second_wave_trend'].tolist()[:-1]
d_working['stagger_drug_resistant'] = [0] + d_working['google_drug_resistant_trend'].tolist()[:-1]




d_working.head()

## Functions to summarize and plot model plots ##


def ols_model_summarize(formula_str, data, cov_type, cov_kwds):
    model = sm.ols(formula=formula_str, data=data)
    model_fit = model.fit(cov_type='cluster',cov_kwds=cov_kwds, use_t=True)


    # R style linear model plots via https://medium.com/@emredjan/emulating-r-regression-plots-in-python-43741952c034

    ###############Calculations required for some of the plots################
    # fitted values (need a constant term for intercept)
    model_fitted_y = model_fit.fittedvalues
    # model residuals
    model_residuals = model_fit.resid
    # normalized residuals
    model_norm_residuals = model_fit.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)
    # leverage, from statsmodels internals
    model_leverage = model_fit.get_influence().hat_matrix_diag
    # cook's distance, from statsmodels internals
    model_cooks = model_fit.get_influence().cooks_distance[0]



    ###########Residual Plot######################
    plot_lm_1 = plt.figure(1)
    plot_lm_1.set_figheight(4)
    plot_lm_1.set_figwidth(6)
    plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'amount_used', data=data, 
                              lowess=True, 
                              scatter_kws={'alpha': 0.5}, 
                              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_lm_1.axes[0].set_title('Residuals vs Fitted')
    plot_lm_1.axes[0].set_xlabel('Fitted values')
    plot_lm_1.axes[0].set_ylabel('Residuals')

    # annotations
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]
    for i in abs_resid_top_3.index:
        plot_lm_1.axes[0].annotate(i, 
                                   xy=(model_fitted_y[i], 
                                       model_residuals[i]));
    
    ######### QQplot#############################
    QQ = ProbPlot(model_norm_residuals)
    plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
    plot_lm_2.set_figheight(4)
    plot_lm_2.set_figwidth(6)
    plot_lm_2.axes[0].set_title('Normal Q-Q')
    plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
    plot_lm_2.axes[0].set_ylabel('Standardized Residuals');
    # annotations
    abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]
    for r, i in enumerate(abs_norm_resid_top_3):
        plot_lm_2.axes[0].annotate(i, 
                                   xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                       model_norm_residuals[i]));
    

    ############ Scale location plot ###############
    plot_lm_3 = plt.figure(3)
    plot_lm_3.set_figheight(4)
    plot_lm_3.set_figwidth(6)
    plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 
                scatter=False, 
                ci=False, 
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_lm_3.axes[0].set_title('Scale-Location')
    plot_lm_3.axes[0].set_xlabel('Fitted values')
    plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');
    # annotations
    abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
    abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
    for i in abs_norm_resid_top_3:
        plot_lm_3.axes[0].annotate(i, 
                                   xy=(model_fitted_y[i], 
                                       model_norm_residuals_abs_sqrt[i]));
    

    ######### Leverage Plot ##################
    plot_lm_4 = plt.figure(4)
    plot_lm_4.set_figheight(4)
    plot_lm_4.set_figwidth(6)
    plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
    sns.regplot(model_leverage, model_norm_residuals, 
                scatter=False, 
                ci=False, 
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_lm_4.axes[0].set_xlim(0, 0.20)
    plot_lm_4.axes[0].set_ylim(-3, 5)
    plot_lm_4.axes[0].set_title('Residuals vs Leverage')
    plot_lm_4.axes[0].set_xlabel('Leverage')
    plot_lm_4.axes[0].set_ylabel('Standardized Residuals')
    # annotations
    leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
    for i in leverage_top_3:
        plot_lm_4.axes[0].annotate(i, 
                                   xy=(model_leverage[i], 
                                       model_norm_residuals[i]))
    
    # shenanigans for cook's distance contours
    def graph(formula, x_range, label=None):
        x = x_range
        y = formula(x)
        plt.plot(x, y, label=label, lw=1, ls='--', color='red')
    p = len(model_fit.params) # number of model parameters
    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
          np.linspace(0.001, 0.200, 50), 
          'Cook\'s distance') # 0.5 line
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
          np.linspace(0.001, 0.200, 50)) # 1 line
    plt.legend(loc='upper right');

    return model_fit.summary()


def mixedlm_model_summarize(formula_str, data, groups):
    model = sm.mixedlm(formula=formula_str, data=data, groups=groups)
    model_fit = model.fit()
    
        # R style linear model plots via https://medium.com/@emredjan/emulating-r-regression-plots-in-python-43741952c034

    ###############Calculations required for some of the plots################
    # fitted values (need a constant term for intercept)
    model_fitted_y = model_fit.fittedvalues
    # model residuals
    model_residuals = model_fit.resid
#     # normalized residuals
#     model_norm_residuals = model_fit.get_influence().resid_studentized_internal
#     # absolute squared normalized residuals
#     model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)
#     # leverage, from statsmodels internals
#     model_leverage = model_fit.get_influence().hat_matrix_diag
#     # cook's distance, from statsmodels internals
#     model_cooks = model_fit.get_influence().cooks_distance[0]



    ###########Residual Plot######################
    plot_lm_1 = plt.figure(1)
    plot_lm_1.set_figheight(4)
    plot_lm_1.set_figwidth(6)
    plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'amount_used', data=data, 
                              lowess=True, 
                              scatter_kws={'alpha': 0.5}, 
                              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_lm_1.axes[0].set_title('Residuals vs Fitted')
    plot_lm_1.axes[0].set_xlabel('Fitted values')
    plot_lm_1.axes[0].set_ylabel('Residuals')

    # annotations
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]
    for i in abs_resid_top_3.index:
        plot_lm_1.axes[0].annotate(i, 
                                   xy=(model_fitted_y[i], 
                                       model_residuals[i]));
    


    return model_fit.summary()





#Model clustered on location and week number###

lm_simple = 'amount_used ~ Treatment'
cov_keys_weeksloc={'groups': [d_working['numeric_location'],d_working['week_number']]}


ols_model_summarize(formula_str=lm_simple, data=d_working, cov_type='cluster', cov_kwds=cov_keys_weeksloc)

#Model clustered on location + stagger_second_flu_trend and week number###

lm_simple_flu = 'amount_used ~ Treatment + stagger_second_flu_trend'
cov_keys_weeksloc={'groups': [d_working['numeric_location'],d_working['week_number']]}


ols_model_summarize(formula_str=lm_simple_flu, data=d_working, cov_type='cluster', cov_kwds=cov_keys_weeksloc)

#Model with dummy location variables, clustered on week number###


lm_locs = 'amount_used ~ Treatment +                  loc_0 + loc_1 + loc_2 + loc_3 + loc_4 +                  loc_5 + loc_6 + loc_7 + loc_8 + loc_9 +                  loc_10'
cov_keys_weeks={'groups': [d_working['week_number']]}


ols_model_summarize(formula_str=lm_locs, data=d_working, cov_type='cluster', cov_kwds=cov_keys_weeks)

#Model with dummy location variables + google_second_wave_trend, clustered on week number###


lm_locs_flu = 'amount_used ~ Treatment + stagger_second_flu_trend +                  loc_0 + loc_1 + loc_2 + loc_3 + loc_4 +                  loc_5 + loc_6 + loc_7 + loc_8 + loc_9 +                  loc_10'
cov_keys_weeks={'groups': [d_working['week_number']]}


ols_model_summarize(formula_str=lm_locs_flu, data=d_working, cov_type='cluster', cov_kwds=cov_keys_weeks)



#Mixed effects model, random intercepts

mm_loc_group = d_working['numeric_location']

mixedlm_model_summarize(formula_str=lm_simple, data=d_working, groups=mm_loc_group)

d_working_2 = d_working[d_working['is_suspicious'] == 0]

#Model clustered on location and week number###
#lm_simple = 'amount_used ~ Treatment'
cov_keys_weeksloc={'groups': [d_working_2['numeric_location'],d_working_2['week_number']]}


ols_model_summarize(formula_str=lm_simple, data=d_working_2, cov_type='cluster', cov_kwds=cov_keys_weeksloc)

#Model with dummy location variables, clustered on week number###


# lm_locs = 'amount_used ~ Treatment + \
#                  loc_0 + loc_1 + loc_2 + loc_3 + loc_4 + \
#                  loc_5 + loc_6 + loc_7 + loc_8 + loc_9 + \
#                  loc_10'
cov_keys_weeks={'groups': [d_working_2['week_number']]}


ols_model_summarize(formula_str=lm_locs, data=d_working_2, cov_type='cluster', cov_kwds=cov_keys_weeks)

#Mixed effects model, random intercepts

mm_loc_group = d_working_2['numeric_location']

mixedlm_model_summarize(formula_str=lm_simple, data=d_working_2, groups=mm_loc_group)



