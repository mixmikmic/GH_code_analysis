get_ipython().magic('matplotlib inline')
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.arima_process import arma_generate_sample
x = arma_generate_sample([1.0, -0.6, 1.0, -0.6], [1.0], 200, 1.0, burnin=100)
plt.plot(x)

f = plt.figure()
ax = f.add_subplot(2,1,1); _ = sm.graphics.tsa.plot_acf(x, lags=100, ax=ax)
ax2 = f.add_subplot(2,1,2); _ = sm.graphics.tsa.plot_pacf(x, lags=100, ax=ax2)


model = sm.tsa.ARMA(x, (3, 1))
result = model.fit(maxiter=1000, method='mle', solver='cg')
result.summary()

_ = result.plot_predict()

plt.plot(result.resid)

_ = result.plot_predict(start=80, plot_insample=True)

f, s, e= result.forecast(steps=50)

plt.plot(f, '.-', e, '.--')



