import numpy as np
import pymc3 as pm
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as graph
from IPython.display import display, Markdown

graph.style.use('fivethirtyeight')

df = pd.read_csv(
    'https://raw.githubusercontent.com/SKKSaikia/MicrosoftLearning-Data.Science.Essentials/master/Mod3/GaltonFamilies.csv',
    index_col=0
)
df['isMale'] = (df['gender'] == 'male').astype(int)

print(df.shape)
display(df.head())

print(f'Number of families = {len(df.family.unique())}')

sns.jointplot(df['midparentHeight'], df['childHeight'], kind='reg')
graph.show()

sns.pairplot(data=df[['midparentHeight', 'childHeight', 'gender']], hue='gender', diag_kind='kde')
graph.show()

df.columns

get_ipython().run_cell_magic('time', '', "# Frequentist GLM\nfglm = smf.glm('childHeight ~ father + mother + C(gender)', data=df).fit()\n\ndisplay(fglm.summary())\n\ngraph.title(f'Residuals $\\sigma =$ {fglm.resid_pearson.std():0.3f} inches')\nsns.distplot(fglm.resid_pearson, fit=stats.norm, kde=False)\ngraph.show()")

get_ipython().run_cell_magic('time', '', "with pm.Model() as bayes_regression:\n    # Priors Bayesian GLM\n    # childHeight ~ father + mother + C(gender)\n    error = pm.Uniform('error', lower=0, upper=25)\n    b = pm.Normal('b', mu=0, sd=10)\n    father = pm.Normal('father', mu=0, sd=1)\n    mother = pm.Normal('mother', mu=0, sd=1)\n    male = pm.Normal('male', mu=0, sd=10)\n    \n    # Model\n    obs = pm.Normal(\n        'y', \n        mu=(father * df['father']) + (mother * df['mother']) + (male * df['isMale']) + b, sd=error,\n        observed=df['childHeight']\n    )\n    \n    # Sample\n    trace = pm.sample(int(10e3), init='jitter+adapt_diag', n_init=1000, discard_tuned_samples=False)\n    \n    # Posterior\n    pm.traceplot(trace)\n    graph.show()\n    \n    trace = trace[1000::3]  # Verified with autocorrelation plots\n    \n    pm.plot_posterior(trace)\n    graph.show()")

# Predictive Posterior Checks
graph.figure(figsize=(8, 8))
for row in zip(df['childHeight'], pm.sample_ppc(trace, samples=100, model=bayes_regression)['y'].T):
    graph.plot(row[0] * np.ones(row[1].shape), row[1], 'o', markersize=1, color='seagreen', alpha=0.5)
graph.xlabel('True Height')
graph.ylabel('Predicted Height')
graph.show()

def create_point(mom, dad, gender):
    p = pd.DataFrame(
        {'mother': mom, 'father': dad, 'gender': 'female' if 'f' in gender else 'male'},
        index=[0]
    )
    p['isMale'] = (p['gender'] == 'male').astype(int)
    return p

def point_estimate(mom, dad, gender):
    """Frequentist single point estimate"""
    point = create_point(mom, dad, gender)
    return fglm.predict(point)[0]

display(Markdown(f'Annie estimated at {point_estimate(71, 67, "f"):0.2f} inches'))
display(Markdown(f'Anthony estimated at {point_estimate(62, 67, "m"):0.2f} inches'))

# Posterior parameters
trace_df = pm.trace_to_dataframe(trace)
display(trace_df.head(3))

def bayes_estimate(mom, dad, gender):
    point = create_point(mom, dad, gender)
    est = []
    for _, row in trace_df.iterrows():
        p = ((row['father'] * point['father']) + (row['mother'] * point['mother']) + (row['male'] * point['isMale']) + row['b'])
        p += stats.norm.rvs(loc=0, scale=row['error'], size=1)
        est.append(p)
    return np.array(est).flatten()

# Test Values
annie = bayes_estimate(71, 67, 'f')
graph.title(f'Annie estimated at {annie.mean():0.2f} inches, $\sigma$ {annie.std():0.2f}')
sns.distplot(annie)
graph.show()

anthony = bayes_estimate(62, 67, 'm')
graph.title(f'Anthony {anthony.mean():0.2f} inches, $\sigma$ {anthony.std():0.2f}')
sns.distplot(anthony)
graph.show()

test_point = bayes_estimate(71, 67, 'm')
graph.title(f'$\mu$ = {test_point.mean():0.2f}')
sns.distplot(test_point)
graph.show()

