get_ipython().magic('pylab inline')
import pandas as pd
import pysd
import scipy.optimize

model = pysd.read_vensim('../../models/Epidemic/SI Model.mdl')

data = pd.read_csv('../../data/Ebola/Ebola_in_SL_Data.csv', index_col='Weeks')
data.plot();

def error(param_list):
    #unpack the parameter list 
    population, contact_frequency = param_list
    #run the model with the new parameters, returning the info we're interested in
    result = model.run(params={'total_population':population,
                               'contact_frequency':contact_frequency},
                       return_columns=['population_infected_with_ebola'],
                       return_timestamps=list(data.index.values))
    #return the sum of the squared errors
    return sum((result['population_infected_with_ebola'] - data['Cumulative Cases'])**2)

error([10000, 10])

susceptible_population_guess = 9000
contact_frequency_guess = 20

susceptible_population_bounds = (2, 50000)
contact_frequency_bounds = (0.001, 100)

res = scipy.optimize.minimize(error, [susceptible_population_guess,
                                      contact_frequency_guess],
                              method='L-BFGS-B',
                              bounds=[susceptible_population_bounds,
                                      contact_frequency_bounds])
res

population, contact_frequency = res.x
result = model.run(params={'total_population':population,
                           'contact_frequency':contact_frequency},
                   return_columns=['population_infected_with_ebola'],
                   return_timestamps=list(data.index.values))

plt.plot(result.index, result['population_infected_with_ebola'], label='Simulated')
plt.plot(data.index, data['Cumulative Cases'], label='Historical');
plt.xlabel('Time [Days]')
plt.ylabel('Cumulative Infections')
plt.title('Model fit to Sierra Leone Ebola historical infections data')
plt.legend(loc='lower right')
plt.text(2,9000, 'RMSE: 7.5% of Max', color='r', fontsize=12)

res

sqrt(res.fun/len(data))/data['Cumulative Cases'].max()



