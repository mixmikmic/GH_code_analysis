get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import pyflux as pf

from statsmodels.tsa.arima_process import arma_generate_sample
xs = arma_generate_sample([1.0, -0.6, 1.0, -0.6], [1.0, 0.5], 200, 1.0, burnin=100)
plt.plot(xs)

model = pf.ARIMA(data=xs, ar=3, ma=2)

result = model.fit()

result.summary()

model.plot_z()

model.plot_fit()

model.plot_predict(h=20)



