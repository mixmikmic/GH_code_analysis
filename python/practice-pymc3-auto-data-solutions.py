import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('fivethirtyeight')

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

auto = pd.read_csv('./datasets/Auto.csv')

auto.isnull().sum()

auto.describe()

auto.head()

print auto.horsepower.unique()
auto.horsepower = auto.horsepower.map(lambda x: np.nan if x == '?' else float(x))
auto.isnull().sum()

auto.dropna(inplace=True)

auto['maker'] = auto.name.map(lambda x: x.split()[0])
auto['maker'].value_counts()

auto.maker.unique()

american = ['chevrolet','buick','plymouth','amc','ford','pontiac','dodge',
            'chevy','oldsmobile','chrysler','chevroelt','cadillac','triumph']
euro = ['volkswagen','peugeot','audi','saab','bmw','fiat','volvo','renualt','vw',
        'mercedes-benz','mercedes','vokswagen']
asian = ['toyota','datsum','mazda','toyouta','maxda','honda','subaru','nissan']

auto['american'] = auto.maker.map(lambda x: 1 if x in american else 0)
auto['euro'] = auto.maker.map(lambda x: 1 if x in euro else 0)
auto['asian'] = auto.maker.map(lambda x: 1 if x in asian else 0)

auto.describe()

sns.pairplot(auto[['mpg','cylinders','displacement','weight','acceleration','year','origin']], hue='origin')

american_mpg = auto[auto.american == 1].mpg.values
european_mpg = auto[auto.euro == 1].mpg.values

print american_mpg.mean(), european_mpg.mean()

import pymc3 as pm

prior_mean = auto.mpg.mean()
prior_std = auto.mpg.std()
print prior_mean, prior_std

with pm.Model() as model:
    
    usa_mean = pm.Normal('usa_mean', prior_mean, sd=50)
    euro_mean = pm.Normal('euro_mean', prior_mean, sd=50)
    
    usa_std = pm.Gamma('usa_std', mu=prior_std, sd=50)
    euro_std = pm.Gamma('euro_std', mu=prior_std, sd=50)
    
    usa_mpg = pm.Normal('usa_mpg', mu=usa_mean, sd=usa_std, observed=american_mpg)
    euro_mpg = pm.Normal('euro_mpg', mu=euro_mean, sd=euro_std, observed=european_mpg)
    
    mean_delta = pm.Deterministic('mean_delta', usa_mean - euro_mean)
    std_delta = pm.Deterministic('std_delta', usa_std - euro_std)
    effect_size = pm.Deterministic('effect_size', mean_delta / np.sqrt((usa_std**2 + euro_std**2)/2.))
    

with model:
    step = pm.NUTS()
    start = pm.find_MAP()
    trace = pm.sample(50000, start=start, step=step, njobs=4)

pm.plot_posterior(trace[5000::3],
                  varnames=['usa_mean', 'euro_mean',
                            'usa_std', 'euro_std'],
                  color='#87ceeb')

pm.plot_posterior(trace[5000::3],
                  varnames=['mean_delta','std_delta','effect_size'],
                  color='#87ceeb', ref_val=0)

auto.columns

auto.mpg.mean()

X = auto[['cylinders','displacement','horsepower','weight','acceleration','year','american','euro','asian']]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Xs = ss.fit_transform(X)
Xs = pd.DataFrame(Xs, columns=X.columns)

with pm.Model() as reg:
    mpg_std = pm.Uniform('error_std', lower=0.0001, upper=auto.mpg.var())
    
    intercept = pm.Normal('intercept', mu=0., sd=100.)
    cyl_beta = pm.Normal('cyl_beta', mu=0., sd=auto.mpg.var())
    disp_beta = pm.Normal('disp_beta', mu=0., sd=auto.mpg.var())
    horse_beta = pm.Normal('horse_beta', mu=0., sd=auto.mpg.var())
    weight_beta = pm.Normal('weight_beta', mu=0., sd=auto.mpg.var())
    acc_beta = pm.Normal('acc_beta', mu=0., sd=auto.mpg.var())
    year_beta = pm.Normal('year_beta', mu=0., sd=auto.mpg.var())
    usa_beta = pm.Normal('usa_beta', mu=0., sd=auto.mpg.var())
    euro_beta = pm.Normal('euro_beta', mu=0., sd=auto.mpg.var())
    asian_beta = pm.Normal('asian_beta', mu=0., sd=auto.mpg.var())
    
    E_mpg = pm.Normal('y_mean', 
                      mu=(intercept +
                          Xs.cylinders.values * cyl_beta +
                          Xs.displacement.values * disp_beta +
                          Xs.horsepower.values * horse_beta +
                          Xs.weight.values * weight_beta +
                          Xs.acceleration.values * acc_beta +
                          Xs.year.values * year_beta +
                          Xs.american.values * usa_beta + 
                          Xs.euro.values * euro_beta +
                          Xs.asian.values * asian_beta),
                      sd=mpg_std, observed=auto.mpg.values)
    

with reg:
    step = pm.NUTS()
    start = pm.find_MAP()
    trace = pm.sample(50000, step=step, start=start, njobs=4)

plt.figure(figsize=(7,21))
pm.traceplot(trace[5000::3])
plt.tight_layout()



