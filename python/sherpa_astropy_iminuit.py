get_ipython().magic('matplotlib inline')

import warnings
warnings.filterwarnings("ignore")

import numpy as np

import matplotlib.pyplot as plt
plt.style.use(['seaborn-talk', 'ggplot'])

from astropy.stats import gaussian_fwhm_to_sigma

from IPython.display import Image

from astropy.modeling.models import Gaussian1D

def fake_data():    
    # Generate fake data
    np.random.seed(42)
    g1 = Gaussian1D(1, 0, 0.2)
    g2 = Gaussian1D(2.5, 0.5, 0.1)
    x = np.linspace(-1, 1, 200)
    y = g1(x) + g2(x) + np.random.normal(0., 0.2, x.shape)
    return x, y

def show_data(x, y):
    plt.figure(figsize=(11, 6))
    plt.plot(x, y, label='Data')
    plt.xlabel('Position')
    plt.ylabel('Flux')

x, y = fake_data()

Image(filename='files/iminuit_api.png')

from iminuit import Minuit

def gauss(x, amplitude, mean, sigma):
    return amplitude * np.exp(- (x - mean) ** 2 / sigma ** 2)

def model_iminuit(x, amplitude_1, mean_1, sigma_1, amplitude_2, mean_2, sigma_2):
    model = gauss(x, amplitude_1, mean_1, sigma_1)
    model += gauss(x, amplitude_2, mean_2, sigma_2)
    return model

def neg_log_likelihood(amplitude_1, mean_1, sigma_1, amplitude_2, mean_2, sigma_2):
    leastsqr = (y - model_iminuit(x, amplitude_1, mean_1, sigma_1,
                                  amplitude_2, mean_2, sigma_2)) ** 2
    return leastsqr.sum()

par_values_minuit = dict(amplitude_1=1, mean_1=0, sigma_1=gaussian_fwhm_to_sigma,
                        amplitude_2=1, mean_2=0.5, sigma_2=gaussian_fwhm_to_sigma * 0.1)

# Set up fitter
fit_minuit = Minuit(neg_log_likelihood, print_level=0, **par_values_minuit)

# Run fit
result_minuit = fit_minuit.migrad()

# Show result
parameters_minuit = [fit_minuit.values[_] for _ in fit_minuit.parameters]
show_data(x, y)
plt.plot(x, model_iminuit(x, *parameters_minuit), label='Fit iminuit')
plt.legend(loc=0)

Image(filename='files/sherpa_api.png')

from sherpa.data import Data1D
from sherpa.models import Gauss1D
from sherpa.stats import LeastSq
from sherpa.optmethods import LevMar
from sherpa.fit import Fit

# Set up model and fit start parameters
g1_sherpa = Gauss1D('gauss_1')
g1_sherpa.ampl, g1_sherpa.fwhm, g1_sherpa.pos = 1, 1, 0

g2_sherpa = Gauss1D('gauss_2')
g2_sherpa.ampl, g2_sherpa.fwhm, g2_sherpa.pos = 2, 0.1, 0.5

model_sherpa = g1_sherpa + g2_sherpa

# Set up data container
data = Data1D('data', x=x, y=y)

# Set up fitter
fit_sherpa = Fit(data=data, model=model_sherpa, stat=LeastSq(),
                 method=LevMar())

# Run fit
result_sherpa = fit_sherpa.fit()

# Show result
show_data(x, y)
plt.plot(x, model_sherpa(x), label='Fit Sherpa')
plt.legend(loc=0)

Image(filename='files/astropy_api.png')

from astropy.modeling.models import Gaussian1D
from astropy.modeling.fitting import LevMarLSQFitter

# Set up model and fit start parameters
g1_astropy = Gaussian1D(1, 0, gaussian_fwhm_to_sigma * 1)
g2_astropy = Gaussian1D(2, 0.5, gaussian_fwhm_to_sigma * 0.1)

model_astropy = g1_astropy + g2_astropy

# Set up fitter
fit_astropy = LevMarLSQFitter()

# Run fit
result_astropy = fit_astropy(model_astropy, x, y)

# Plot astropy result

show_data(x, y)
plt.plot(x, result_astropy(x), label='Fit Astropy')
plt.legend(loc=0)

