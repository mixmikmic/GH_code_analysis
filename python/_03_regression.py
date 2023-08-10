import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn import preprocessing
import pymc3 as pm
import theano.tensor as ttens


import holoviews as hv
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.layouts import column

hv.notebook_extension('bokeh')
get_ipython().run_line_magic('matplotlib', 'inline')

# Number of data points
N = 100
# Intercept
alpha = 1
# Slope
beta = 2
# Predictor
X = np.linspace(-6,6, N)
# Target
y = alpha + (beta * X) + np.random.normal(0,3,size=N)
# Turn X from a vector into a 1D array
X = X[:,np.newaxis]

fake_scatter = hv.Points((X[:,0],y), group="Scatter plot of fake data")
fake_scatter

def linear_regression(X, y, 
                      alpha_prior, beta_prior, sigma_prior,
                      samples = 1000, njobs = 2):
    # Number of predictors
    K = X.shape[1]
    # Number of data points
    N = X.shape[0]

    with pm.Model() as model: 
        #Specify priors
        alpha = pm.Normal('alpha', mu = alpha_prior['mu'], sd = alpha_prior['sd'])
        beta = pm.Normal('beta', mu = beta_prior['mu'], sd = beta_prior['sd'], shape = int(K))
        sigma = pm.HalfNormal('sigma', sd = sigma_prior['sd'])
        
        # Specify regression relationship in mu
        # Need to use the Theano dot product function to combine beta and X
        mu = alpha + ttens.dot(beta,X.T)
        
        # Specify likelihood
        Y_obs = pm.Normal('Y_obs', mu = mu, sd = sigma, observed = y)
        
        # Fit the model
        trace = pm.sample(draws = samples, njobs = njobs)
        
        # Generate data from the model
        simulate_data = pm.sample_ppc(trace=trace,model=model)
    return model, trace, simulate_data

alpha_prior = {'mu':0, 'sd':1}
beta_prior = {'mu':0, 'sd':2}
sigma_prior = {'sd':2}

model_fake, trace_fake, simulated_data_fake = linear_regression(X,y,
                                                alpha_prior = alpha_prior,
                                                beta_prior = beta_prior,
                                                sigma_prior = sigma_prior,
                                                njobs = 4)

pm.diagnostics.gelman_rubin(trace_fake)

pm.diagnostics.effective_n(trace_fake)

pm.traceplot(trace_fake);

# Get the lines through the data set with no noise
mu = trace_fake.get_values('alpha')[:,np.newaxis] + trace_fake.get_values('beta')*X.T

# Plot the line of best fit as the mean of mu
plt.plot(X[:,0],mu.mean(axis=0));
# Plot the original fake data
plt.scatter(X[:,0],y)

# Create a dataframe df that holds the daily weather data
df = pd.read_csv('data/temp_rain_1960_to_2017.csv',index_col='date', parse_dates=True)
# Select the period that overlaps with the temperature data
trim = df.loc['1962-01-01':'2016-12-31']
# Create a new dataframe with the annual mean weather data
annual = trim.groupby(trim.index.year).mean()
# Print the first five rows of the dataset
annual.head()

# Create a dataframe called maunca that holds the daily CO2 data
# The parse_dates=True means that it interprets the date column in the .csv file as a date
mauna = pd.read_csv('data/mauna_loa.csv',index_col='date',parse_dates=True)
# Create a new dataframe with the annual mean data
annual_mauna = mauna.groupby(mauna.index.year).mean()
# Select the period that overlaps with the temperature data
annual_mauna = annual_mauna.loc['1961-12-31':'2016-12-31']
# Print the first five rows of the annual mean data
annual_mauna.head()

get_ipython().run_cell_magic('output', 'size=150', '%%opts Curve.CO2 (color="red")\nco2_ann = hv.Curve((annual_mauna.index,annual_mauna.co2), \n                      kdims=["Year"],vdims=["Annual mean CO2 concentration (ppm)"],\n                      group="CO2")\ntemp_ann = hv.Curve((annual.index,annual.temp), \n                    kdims=["Year"],vdims=["Temperature degrees C"],\n                   group="Temperature")\n\nco2_ann + temp_ann')

print("Estimated temperature change is {:.2f} degrees per unit CO2 ".format( (annual.temp.iloc[-1]-annual.temp.iloc[0])/(annual_mauna.co2.iloc[-1]-annual_mauna.co2.iloc[0])))

get_ipython().run_cell_magic('output', 'size=150 # Set figure size', '%%opts Scatter (size=7)\n\nco2_temp = hv.Scatter((annual_mauna.co2.values, annual.temp.values),\n                               kdims=["CO2 ppm"], vdims = ["Temperature degrees C"],\n                                                  group = \'Original units\')\n\n# Output the scatter plot\nco2_temp')

# Get the parameters for the centering and standardization
scaler_temp = preprocessing.StandardScaler().fit(annual.temp.values[:,np.newaxis])
# Apply these parameters to a new scaled temperature series
annual['temp_scaled'] = scaler_temp.transform(annual.temp[:,np.newaxis])

scaler_co2 = preprocessing.StandardScaler().fit(annual_mauna.co2.values[:,np.newaxis])
# Apply these parameters to a new scaled temperature series
annual_mauna['co2_scaled'] = scaler_co2.transform(annual_mauna.co2[:,np.newaxis])

get_ipython().run_cell_magic('opts', 'Scatter (size=7)', '\nco2_temp_scaled = hv.Scatter((annual_mauna.co2_scaled.values, annual.temp_scaled.values),\n                               kdims=["CO2 ppm"], vdims = ["Temperature degrees C"],\n                             group = \'scaled\') \nco2_temp_scaled')

co2_temp

# Specify our target vector y and our standardized feature array X
y_train_scaled = annual.temp_scaled.loc[1962:1992].values
y_train = annual.temp.loc[1962:1992].values
y_test = annual.temp_scaled.loc[1993:].values

# X needs to be an array rather than a vector, so we do [:,np.newaxis] to transform it from a vector to a 1D array
X_train_scaled = annual_mauna.co2_scaled.loc[1962:1992].values[:,np.newaxis]
X_train = annual_mauna.co2.loc[1962:1992].values[:,np.newaxis]

X_test = annual_mauna.co2_scaled.loc[1993:].values[:,np.newaxis]

time_train = annual.index[:31]
time_test = annual.index[32:]

alpha_prior = {'mu':0, 'sd':3}
beta_prior = {'mu':0, 'sd':3}
sigma_prior = {'sd':3}

model, trace, simulated_data = linear_regression(X_train_scaled,y_train_scaled,
                                                alpha_prior = alpha_prior,
                                                beta_prior = beta_prior,
                                                sigma_prior = sigma_prior)

pm.diagnostics.gelman_rubin(trace)

pm.diagnostics.effective_n(trace)

pm.traceplot(trace);

alpha = (trace.get_values('alpha')*scaler_temp.scale_) + annual.temp.mean() -(trace.get_values('beta')[:,0]*(scaler_temp.scale_/scaler_co2.scale_)*annual_mauna.co2.mean())

beta = trace.get_values('beta')*(scaler_temp.scale_/scaler_co2.scale_)

print('Mean value of the regression slope is {:.2}'.format(beta.mean()))

sigma = trace.get_values('sigma')
sigma.mean()

get_ipython().run_cell_magic('output', 'size=150', 'mod = hv.Curve((annual.index, alpha.mean() + beta.mean()*annual_mauna.co2.values))\ndata = hv.Curve((annual.index,annual.temp),kdims=["Year"],vdims=["Temperature"])\ndata*mod')

y_train[0]

# Generate the predictions for \mu under the model with shape (4000,55)
mu = alpha[:,np.newaxis] + beta*X_train
# Generate samples from the model
model = np.random.normal(mu,sigma[:,np.newaxis])
# Calculate some statistics from the model predictions
model_mean = model.mean(axis=0)
model_upper = np.percentile(model,97.5,axis=0)
model_lower = np.percentile(model,2.5,axis=0)

def uncertain_prediction(time_index,data, model_mean, model_lower,model_upper, 
                         title = 'Data versus model prediction', ylabel = 'Temperature', xlabel = 'Year'):
    """Make a bokeh plot to show the probability distribution function for rho compared with its true value
    
    Inputs:
    time_index: (1d array) - Time index for x-axis
    data: (1d array) - Observed data
    model_mean (1d array) - Mean of model predictions
    model_lower: (1d array) - Lower bound for temperature in 95% range
    model_upper: (1d array) - Upper bound for temperature in 95% range
    """
    #Have to append a reversed series for the patch coordinates
    band_x = np.append(time_index, time_index[::-1])
    band_y = np.append(model_lower, model_upper[::-1])
    p = figure(title= title, height = 500, width = 900)
    p.line(time_index, data, color = 'blue')
    p.line(time_index, model_mean, color = 'black')
    p.patch(band_x, band_y, color='green', fill_alpha=0.5, line_color = "green")
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel
    show(p)

uncertain_prediction(time_train, y_train,model_mean,model_lower,model_upper, 
                     title="Data and model 95% range")

# Split the data into training and testing data
time_train = annual.index[:30]
time_test = annual.index[30:]

y_train = annual.temp.iloc[:30].values
X_train = annual_mauna.co2_scaled.iloc[:30].values[:,np.newaxis]

y_test = annual.temp.iloc[30:].values
X_test = annual_mauna.co2_scaled.iloc[30:].values[:,np.newaxis]

model2, trace2, simulated_data2 = linear_regression(X_train,y_train)

alpha2 = trace2.get_values('alpha')[:,np.newaxis]  - annual_mauna.co2.mean()*trace2.get_values('beta')/scaler_co2.scale_
beta2 = trace2.get_values('beta')/scaler_co2.scale_
sigma2 = trace2.get_values('sigma')[:,np.newaxis]

# Generate the predictions for \mu under the model with shape (4000,55)
mu_train = alpha2 + np.dot(beta2,X_train.T)
mu_test = alpha2 + np.dot(beta2,X_test.T)

# Generate samples from the model
yfit_train = np.random.normal(mu_train,sigma2)
yfit_test = np.random.normal(mu_test,sigma2)

alpha.mean()

predicted_values = {}
predicted_values['yfit_train'] = yfit_train
predicted_values['yfit_test'] = yfit_test
predicted_values['time_test'] = time_test
predicted_values['time_train'] = time_train

def uncertain_prediction_split(time_train,data_train, time_test, data_test,
                               predicted_values, 
                         title = 'Data versus model prediction', ylabel = 'Temperature', xlabel = 'Year'):
    """Make a bokeh plot to show the probability distribution function for rho compared with its true value
    
    Inputs:
    time_index: (1d array) - Time index for x-axis
    data: (1d array) - Observed data
    model_mean (1d array) - Mean of model predictions
    model_lower: (1d array) - Lower bound for temperature in 95% range
    model_upper: (1d array) - Upper bound for temperature in 95% range
    """
    p = figure(title= title, height = 500, width = 900)
    p.line(time_train, data_train, color = 'blue')
    p.line(time_test, data_test, color = 'blue')
    for key in ['yfit_train','yfit_test']:
        value = predicted_values[key]
        model_mean = value.mean(axis=0)
        model_lower = np.percentile(value,2.5,axis=0)
        model_upper = np.percentile(value,97.5,axis=0)
        #Have to append a reversed series for the patch coordinates
        band_y = np.append(model_lower, model_upper[::-1])
        if 'train' in key:
            band_x = np.append(predicted_values['time_train'], predicted_values['time_train'][::-1])

            p.line(time_train, model_mean, color = 'black')
            p.patch(band_x, band_y, color='green', fill_alpha=0.5, line_color = "green")
        else:
            band_x = np.append(predicted_values['time_test'], predicted_values['time_test'][::-1])
            p.line(time_test, model_mean, color = 'black')
            p.patch(band_x, band_y, color='firebrick', fill_alpha=0.5, line_color = "firebrick")
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel
    show(p)

predicted_values['yfit_test'].min()

uncertain_prediction_split(time_train, y_train, time_test, y_test, predicted_values)

