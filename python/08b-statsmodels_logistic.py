import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

titanic = sns.load_dataset('titanic')
titanic.head()

titanic.info()

model = smf.logit('survived ~ sex + age + fare + deck',
                 data = titanic)
results = model.fit()
results.summary()

# interpret results
np.exp(results.params)

np.exp(results.conf_int())



