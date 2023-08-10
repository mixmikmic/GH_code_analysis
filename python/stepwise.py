# code from http://planspace.org/20150423-forward_selection_with_statsmodels/

import statsmodels.formula.api as smf

def forward_selection(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)  # remove dependent variable
    selected = []  # to hold selected independent variables
    current_score, best_new_score = 0.0, 0.0  # set scores to 0 before iterations
    while remaining and current_score == best_new_score:  # while there are still independent vars to test
        scores_with_candidates = []
        for candidate in remaining:  # each possible ind. var.
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))  # add to already selected ind. vars
            
            score = smf.ols(formula, data).fit().rsquared_adj  # run the reg. and get the adj. rsquared
            scores_with_candidates.append((score, candidate))  # append the adj. rsquared and ind. var. name
        scores_with_candidates.sort()  # sort scores low to high
        best_new_score, best_candidate = scores_with_candidates.pop()  # assign and remove highest score and name
        if current_score < best_new_score:  # if the new score is better than the old
            remaining.remove(best_candidate)  # remove ind. var. from remaining
            selected.append(best_candidate)  # add ind. var. to final selection
            current_score = best_new_score  # make this score the new one to beat
            
    # if all variables were tested or the score did not improve
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))  # format the formula string for smf
    model = smf.ols(formula, data).fit()  # fit and return the final model
    return model

import pandas as pd

url = "http://data.princeton.edu/wws509/datasets/salary.dat"
data = pd.read_csv(url, sep='\\s+')

data

model = forward_selection(data, 'sl')

model.model.formula

model.rsquared_adj

model.summary()

import numpy as np

cols_to_transform = ["sx", "rk", "dg"]
df_with_dummies = pd.get_dummies(data, columns = cols_to_transform )
np.array(df_with_dummies.drop("sl", 1))
print(df_with_dummies.drop("sl", 1))

from sklearn.feature_selection import f_regression
f_regression(np.array(df_with_dummies.drop("sl", 1)), np.array(data["sl"]), center=True)



