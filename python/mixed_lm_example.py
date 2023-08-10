get_ipython().magic('matplotlib inline')

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

get_ipython().magic('load_ext rpy2.ipython')

get_ipython().magic('R library(lme4)')

data = sm.datasets.get_rdataset('dietox', 'geepack').data
md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])
mdf = md.fit()
print(mdf.summary())

get_ipython().run_cell_magic('R', '', "data(dietox, package='geepack')")

get_ipython().magic("R print(summary(lmer('Weight ~ Time + (1|Pig)', data=dietox)))")

md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"], re_formula="~Time")
mdf = md.fit()
print(mdf.summary())

get_ipython().magic('R print(summary(lmer("Weight ~ Time + (1 + Time | Pig)", data=dietox)))')

.294 / (19.493 * .416)**.5

md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"],
                  re_formula="~Time")
free = sm.regression.mixed_linear_model.MixedLMParams.from_components(np.ones(2), 
                                                                      np.eye(2))

mdf = md.fit(free=free)
print(mdf.summary())

get_ipython().magic('R print(summary(lmer("Weight ~ Time + (1 | Pig) + (0 + Time | Pig)", data=dietox)))')

data = sm.datasets.get_rdataset("Sitka", "MASS").data
endog = data["size"]
data["Intercept"] = 1
exog = data[["Intercept", "Time"]]

md = sm.MixedLM(endog, exog, groups=data["tree"], exog_re=exog["Intercept"])
mdf = md.fit()
print(mdf.summary())

get_ipython().run_cell_magic('R', '', 'data(Sitka, package="MASS")\nprint(summary(lmer("size ~ Time + (1 | tree)", data=Sitka)))')

get_ipython().magic('R print(summary(lmer("size ~ Time + (1 + Time | tree)", data=Sitka)))')

exog_re = exog.copy()
md = sm.MixedLM(endog, exog, data["tree"], exog_re)
mdf = md.fit()
print(mdf.summary())

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    likev = mdf.profile_re(0, 're', dist_low=0.1, dist_high=0.1)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plt.plot(likev[:,0], 2*likev[:,1])
plt.xlabel("Variance of random slope", size=17)
plt.ylabel("-2 times profile log likelihood", size=17)

re = mdf.cov_re.iloc[1, 1]
likev = mdf.profile_re(1, 're', dist_low=.5*re, dist_high=0.8*re)

plt.figure(figsize=(10, 8))
plt.plot(likev[:,0], 2*likev[:,1])
plt.xlabel("Variance of random slope", size=17)
plt.ylabel("-2 times profile log likelihood", size=17)



