get_ipython().magic('pylab inline')
import pandas as pd
import pysd
import scipy.optimize
import geopandas as gpd

data = pd.read_csv('../../data/Census/Males by decade and county.csv', header=[0,1], index_col=[0,1])
data.head()

model = pysd.read_vensim('../../models/Aging_Chain/Aging_Chain.mdl')

model.run().plot();

state_geo = gpd.read_file('../../data/Census/US_State.shp')
state_geo.set_index('StateFIPSN', inplace=True)
state_geo.plot();
state_geo.head(2)

country = data.sum()
country

model.run(return_timestamps=range(2000,2011), 
          initial_condition=(2000, country['2000'])).plot();

def exec_model(paramlist):
    params = dict(zip(['dec_%i_loss_rate'%i for i in range(1,10)], paramlist)) 
    output = model.run(initial_condition=(2000,country['2000']),
                       params=params, return_timestamps=2010)
    return output

def error(paramlist):
    output = exec_model(paramlist)
    errors = output - country['2010']
    #don't tally errors in the first cohort, as we don't have info about births
    return sum(errors.values[0,1:]**2)

res = scipy.optimize.minimize(error, x0=[.05]*9,
                              method='L-BFGS-B')
country_level_fit_params = dict(zip(['dec_%i_loss_rate'%i for i in range(1,10)], res['x']))
country_level_fit_params

model.run(params=country_level_fit_params,
          return_timestamps=range(2000,2011), 
          initial_condition=(2000, country['2000'])).plot();

states = data.sum(level=0)
states.head()

def model_runner(row):
    result = model.run(params=country_level_fit_params, 
                       initial_condition=(2000,row['2000']), 
                       return_timestamps=2010)
    return result.loc[2010]
    
state_predictions = states.apply(model_runner, axis=1)
state_predictions.head()

diff = state_predictions-states['2010']
diff.head()

diff_percent = (state_predictions-states['2010'])/states['2010']
diff_percent.head()

geo_diff = state_geo.join(diff_percent)
geo_diff.plot(column='dec_4')
geo_diff.head()

def exec_model(paramlist, state):
    params = dict(zip(['dec_%i_loss_rate'%i for i in range(1,10)], paramlist)) 
    output = model.run(initial_condition=(2000,state['2000']),
                       params=params, return_timestamps=2010).loc[2010]
    return output

def error(paramlist, state):
    output = exec_model(paramlist, state)
    errors = output - state['2010']
    #don't tally errors in the first cohort, as we don't have info about births
    sse = sum(errors.values[1:]**2)
    return sse

get_ipython().run_cell_magic('capture', '', "def optimize_params(row):\n    res = scipy.optimize.minimize(lambda x: error(x, row),\n                                  x0=[.05]*9,\n                                  method='L-BFGS-B');\n    return pd.Series(index=['dec_%i_loss_rate'%i for i in range(1,10)], data=res['x'])\n    \nstate_fit_params = states.apply(optimize_params, axis=1)\nstate_fit_params.head()")

geo_diff = state_geo.join(state_fit_params)
geo_diff.plot(column='dec_4_loss_rate')
geo_diff.head(3)



