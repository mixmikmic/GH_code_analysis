get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels as sm
import statsmodels.api as sma
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

def estimate_propensity(formula, data, show_summary=False):
    """ Estimates a propensity model using generalized linear models.
    
    Arguments:
    ----------
        formula:       String that contains the (R style) model formula.
        data:          DataFrame containing the covariates for each observation.
        show_summary:  Boolean that indicates to print out the model summary to the console.
    """
    model = smf.glm(formula=formula, data=data, family=sma.families.Binomial()).fit()

    if show_summary:
        print(model.summary())

    return pd.Series(data=model.predict(data), index=data.index)


def matching(label, propensity, calipher=0.05, replace=True):
    """ Performs nearest-neighbour matching for a sample of test and control
    observations, based on the propensity scores for each observation.

    Arguments:
    ----------
        label:      Series that contains the label for each observation.
        propensity: Series that contains the propensity score for each observation.
        calipher:   Bound on distance between observations in terms of propensity score.
        replace:    Boolean that indicates whether sampling is with (True) or without replacement (False).
    """

    treated = propensity[label == 1]
    control = propensity[label == 0]

    # Randomly permute in case of sampling without replacement to remove any bias arising from the
    # ordering of the data set
    matching_order = np.random.permutation(label[label == 1].index)
    matches = {}

    for obs in matching_order:
        # Compute the distance between the treatment observation and all candidate controls in terms of
        # propensity score
        distance = abs(treated[obs] - control)

        # Take the closest match
        if distance.min() <= calipher or not calipher:
            matches[obs] = [distance.argmin()]
            
            # Remove the matched control from the set of candidate controls in case of sampling without replacement
            if not replace:
                control = control.drop(matches[obs])

    return matches

                            
def matching_to_dataframe(match, covariates, remove_duplicates=False):
    """ Converts a list of matches obtained from matching() to a DataFrame.
    Duplicate rows are controls that where matched multiple times.

    Arguments:
    ----------
        match:              Dictionary with a list of matched control observations.
        covariates:         DataFrame that contains the covariates for the observations.
        remove_duplicates:  Boolean that indicates whether or not to remove duplicate rows from the result. 
                            If matching with replacement was used you should set this to False****
    """
    treated = list(match.keys())
    control = [ctrl for matched_list in match.values() for ctrl in matched_list]

    result = pd.concat([covariates.loc[treated], covariates.loc[control]])

    if remove_duplicates:
        return result.groupby(result.index).first()
    else:
        return result
     

def trim_common_support(data, label_name):
    """ Removes observations that fall outside the common support of the propensity score 
        distribution from the data.
    
    Arguments:
    ----------
        data:        DataFrame with the propensity scores for each observation.
        label_name:  Column name that contains the labels (treatment/control) for each observation.
    
    """
    group_min_max = (data.groupby(label_name)
                         .propensity.agg({"min_propensity": np.min, "max_propensity": np.max}))

    # Compute boundaries of common support between the two propensity score distributions
    min_common_support = np.max(group_min_max.min_propensity)
    max_common_support = np.min(group_min_max.max_propensity)

    common_support = (data.propensity >= min_common_support) & (data.propensity <= max_common_support)
    control = (data[label_name] == 0)
    treated = (data[label_name] == 1)
    
    return data[common_support]

def qq_plot(x, y, variable_name):
    """ Produces a QQ-plot where the percentiles of two empirical distributions are compared against each other.
    
    Arguments:
    ----------
        x:              Vector with samples from the first distribution.
        y:              Vector with samples from the second distribution.
        variable_name:  Name of the variable.
    
    """
    q = np.arange(0, 100)
    a = np.percentile(a=x, q=q)
    b = np.percentile(a=y, q=q)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.title("QQ-plot of variable %s" % variable_name)
        
    plt.plot(a, b, 'o')
    plt.plot([0, np.max(a)], [0, np.max(a)], '-')
    
    ax.set_xlabel("Control quantiles")
    ax.set_ylabel("Treatment quantiles")

# Import the LaLonde data set
lalonde = pd.read_csv("lalonde.csv", index_col=0)
lalonde.head()

lalonde.groupby("treat")["re74"].plot(kind="hist", sharex=True, range=(0, 40000), bins=30, alpha=0.75)

lalonde.groupby("treat")["re75"].plot(kind="hist", sharex=True, range=(0, 40000), bins=30, alpha=0.75)

propensity_model_covariates = ['age', 'educ', 'black', 'hispan', 
                               'married', 'nodegree', 're74', 're75',
                               'I(age^2)', 'I(educ^2)', 'I(re74 ** 2)', 'I(re75 ** 2)', 're74 * black']

propensity_model_formula = "treat ~ " + " + ".join(propensity_model_covariates)

lalonde["propensity"] = estimate_propensity(formula=propensity_model_formula, 
                                            data=lalonde, 
                                            show_summary=True)

lalonde.groupby("treat")["propensity"].plot(kind="hist", sharex=True, range=(0, 1), bins=20, alpha = 0.75)

common_support = trim_common_support(lalonde, "treat")

# Plot the resulting propensity score distribution to make sure we restricted ourselves to the common support
common_support.groupby("treat")["propensity"].plot(kind="hist", sharex=True, range=(0, 1), bins=20, alpha = 0.75)

matches = matching(label=common_support.treat,
                   propensity=common_support.propensity,
                   calipher=0.01,
                   replace=True)

# Did everybody get a match?
sum([True if match == [] else False for match in matches]) == 0

matches_data_frame = matching_to_dataframe(match=matches,
                                           covariates=common_support,
                                           remove_duplicates=False)

matches_data_frame.groupby("treat")["propensity"].plot(kind="hist", sharex=True, range=(0,1), bins=20, alpha=0.75)

vars_to_compare = ["propensity", "re74", "re75"]
matches_data_frame.groupby("treat")[vars_to_compare].agg([np.mean, np.std])

vars_to_plot = ["re74", "re75"]

for var in vars_to_plot:
    control = matches_data_frame[matches_data_frame.treat == 0][var]
    treat = matches_data_frame[matches_data_frame.treat == 1][var]
    
    qq_plot(control, treat, var)

def loess_plots(matches_data_frame, balance_covariates, label_name):
    for covariate in balance_covariates:
        plt.figure()
        sns.lmplot(data=matches_data_frame,
                   x="propensity",
                   y=covariate,
                   hue=label_name,
                   scatter=False, 
                   lowess=True)
        
loess_plots(matches_data_frame, vars_to_plot, label_name="treat")

vars_to_plot = ["propensity", "re74", "re75"]
vars_range = {"propensity": (0, 1),
              "re74": (0, 40000),
              "re75": (0, 40000)}

for var in vars_to_plot:
    fig, ax = plt.subplots()
    matches_data_frame.groupby("treat")[var].plot(kind="hist", sharex=True, range=vars_range[var], bins=15, alpha=0.75)

print(smf.ols(formula='re78 ~ treat', data=matches_data_frame).fit().summary())

