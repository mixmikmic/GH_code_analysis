get_ipython().magic('matplotlib inline')
np.random.seed(20090425)
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pymc3 as pm
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
from scipy.stats import ttest_ind
import sys
sys.path.append('../ThinkStats2/code')
from thinkstats2 import HypothesisTest

def get_col_vals(df, col1, col2):
    """
    Get column values 
    """
    y1 = np.array(df[col1])
    y2 = np.array(df[col2])
    return y1, y2

def prep_data(df, col1, col2):
    """
    Prepare data for pymc3 and return mean mu and sigma
    """
    y1 = np.array(df[col1])
    y2 = np.array(df[col2])

    y = pd.DataFrame(dict(value=np.r_[y1, y2], 
                          group=np.r_[[col1]*len(y1), 
                            [col2]*len(y2)]))
    mu = y.value.mean()
    sigma = y.value.std() * 2
    
    return y, mu, sigma

class DiffMeansPermute(HypothesisTest):
    """
    Model the null hypothesis, which says that the distributions
    for the two groups are the same.
    data: pair of sequences (one for each group)
    """
    def TestStatistic(self, data):
        """
        Calculate the test statistic, the absolute difference in means
        """
        group1, group2 = data
        test_stat = abs(np.mean(group1) - np.mean(group2))
        return test_stat

    def MakeModel(self):
        """
        Record the sizes of the groups, n and m, 
        and combine into one Numpy array, self.pool
        """
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        
        # make group1 and group2 into a single array
        self.pool = np.concatenate((group1, group2))

    def RunModel(self):
        """
        Simulate the null hypothesis- shuffle the pooled values 
        and split into 2 groups with sizes n and m
        """
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data

def Resample(x):
    """
    Get a bootstrap sample
    """
    return np.random.choice(x, len(x), replace=True)

scores = pd.read_excel('test_scores.xlsx')

scores.head()

scores.describe()

y1, y2 = get_col_vals(scores, 'group1', 'group2')
ht = DiffMeansPermute((y1, y2))

pval=ht.PValue()
pval

ttest_ind(y1,y2, equal_var=False)

y, mu, sigma = prep_data(scores, 'group1', 'group2')
y.hist('value', by='group');

μ_m = y.value.mean()
μ_s = y.value.std() * 2

with pm.Model() as model:
    """
    The priors for each group.
    """
    group1_mean = pm.Normal('group1_mean', μ_m, sd=μ_s)
    group2_mean = pm.Normal('group2_mean', μ_m, sd=μ_s)

σ_low = 1
σ_high = 20

with model:
    group1_std = pm.Uniform('group1_std', lower=σ_low, upper=σ_high)
    group2_std = pm.Uniform('group2_std', lower=σ_low, upper=σ_high)

with model:
    """
    Prior for ν is an exponential (lambda=29) shifted +1.
    """
    ν = pm.Exponential('ν_min_one', 1/29.) + 1

sns.distplot(np.random.exponential(30, size=10000), kde=False);

with model:
    """
    Transforming standard deviations to precisions (1/variance) before
    specifying likelihoods.
    """
    λ1 = group1_std**-2
    λ2 = group2_std**-2

    group1 = pm.StudentT('group1', nu=ν, mu=group1_mean, lam=λ1, observed=y1)
    group2 = pm.StudentT('group2', nu=ν, mu=group2_mean, lam=λ2, observed=y2)

with model:
    """
    The effect size is the difference in means/pooled estimates of the standard deviation.
    The Deterministic class represents variables whose values are completely determined
    by the values of their parents.
    """
    diff_of_means = pm.Deterministic('difference of means',  group2_mean - group1_mean)
    diff_of_stds = pm.Deterministic('difference of stds',  group2_std - group1_std)
    effect_size = pm.Deterministic('effect size',
                                   diff_of_means / np.sqrt((group2_std**2 + group1_std**2) / 2))

with model:
    trace = pm.sample(2000, njobs=2)

pm.plot_posterior(trace[1000:],
                  varnames=['group1_mean', 'group2_mean', 'group1_std', 'group2_std', 'ν_min_one'],
                  color='#87ceeb');

pm.plot_posterior(trace[1000:],
                  varnames=['difference of means', 'difference of stds', 'effect size'],
                  ref_val=0,
                  color='#87ceeb');

