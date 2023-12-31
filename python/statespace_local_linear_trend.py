get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt

"""
Univariate Local Linear Trend Model
"""
class LocalLinearTrend(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Model order
        k_states = k_posdef = 2

        # Initialize the statespace
        super(LocalLinearTrend, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef,
            initialization='approximate_diffuse',
            loglikelihood_burn=k_states
        )

        # Initialize the matrices
        self.ssm['design'] = np.array([1, 0])
        self.ssm['transition'] = np.array([[1, 1],
                                       [0, 1]])
        self.ssm['selection'] = np.eye(k_states)

        # Cache some indices
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ['sigma2.measurement', 'sigma2.level', 'sigma2.trend']

    @property
    def start_params(self):
        return [np.std(self.endog)]*3

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super(LocalLinearTrend, self).update(params, *args, **kwargs)
        
        # Observation covariance
        self.ssm['obs_cov',0,0] = params[0]

        # State covariance
        self.ssm[self._state_cov_idx] = params[1:]

import requests
from io import BytesIO
from zipfile import ZipFile
    
# Download the dataset
ck = requests.get('http://staff.feweb.vu.nl/koopman/projects/ckbook/OxCodeAll.zip').content
zipped = ZipFile(BytesIO(ck))
df = pd.read_table(
    BytesIO(zipped.read('OxCodeIntroStateSpaceBook/Chapter_2/NorwayFinland.txt')),
    skiprows=1, header=None, sep='\s+', engine='python',
    names=['date','nf', 'ff']
)

# Load Dataset
df.index = pd.date_range(start='%d-01-01' % df.date[0], end='%d-01-01' % df.iloc[-1, 0], freq='AS')

# Log transform
df['lff'] = np.log(df['ff'])

# Setup the model
mod = LocalLinearTrend(df['lff'])

# Fit it using MLE (recall that we are fitting the three variance parameters)
res = mod.fit()
print(res.summary())

# Perform prediction and forecasting
predict = res.get_prediction()
forecast = res.get_forecast('2014')

fig, ax = plt.subplots(figsize=(10,4))

# Plot the results
df['lff'].plot(ax=ax, style='k.', label='Observations')
predict.predicted_mean.plot(ax=ax, label='One-step-ahead Prediction')
predict_ci = predict.conf_int(alpha=0.05)
predict_index = np.arange(len(predict_ci))
ax.fill_between(predict_index[2:], predict_ci.iloc[2:, 0], predict_ci.iloc[2:, 1], alpha=0.1)

forecast.predicted_mean.plot(ax=ax, style='r', label='Forecast')
forecast_ci = forecast.conf_int()
forecast_index = np.arange(len(predict_ci), len(predict_ci) + len(forecast_ci))
ax.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], alpha=0.1)

# Cleanup the image
ax.set_ylim((4, 8));
legend = ax.legend(loc='lower left');

