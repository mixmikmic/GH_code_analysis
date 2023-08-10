get_ipython().magic('pylab inline')
import pandas as pd
import pysd
import pymc as mc

plt.figure(figsize(6,2))
model = pysd.read_vensim('penny_jar.mdl')
model.run().plot()
#model.get_free_parameters()

production = pd.read_csv('Production_Figures.csv', index_col='Year')
production.plot()
plt.title('Pennies Produced Per Year');

coin_counts = pd.read_csv('pennies_in_jar.csv', index_col='Year')
coin_counts.sort_index().plot()
plt.title('Pennies in my Jar');

coin_counts.sum()

plt.figure(figsize=(12,5))
plt.bar(coin_counts.index, coin_counts['Philadelphia']/sum(coin_counts['Philadelphia']))
plt.title('Predicted Pennies in Jar', fontsize=16)
plt.xlabel('Mint Year', fontsize=16)
plt.ylabel('Likelihood for any given penny', fontsize=16)
plt.xlim(1950,2015)

#load a model for each mint year
models = pd.DataFrame(data=[[year, pysd.read_vensim('penny_jar.mdl')] for year in range(1930,2014)],
                      columns=['Year', 'model'])

models.set_index(keys='Year', drop=False, inplace=True)

#bring in the data on production
models['Philadelphia Production'] = production['Philadelphia'] / 100000 
#production will now be in units of hundred-thousands

#bring in the sample data
models['Philadelphia Samples'] = coin_counts['Philadelphia']

#set the mint year parameters properly
for index, row in models.iterrows():
    row['model'].set_components({'production_year':row['Year'], 'production_volume':row['Philadelphia Production']})

#drop rows (probably at the end) which are missing data
models.dropna(inplace=True) 

models.tail(30).head(5)

entry_rate = mc.Uniform('entry_rate', lower=0, upper=.99, value=.08)
loss_rate = mc.Uniform('loss_rate', lower=0, upper=.3, value=.025)

def get_population(model, entry_rate, loss_rate):
    in_circulation = model.run(params={'entry_rate':entry_rate, 'loss_rate':loss_rate}, 
                               return_columns=['in_circulation'],
                               return_timestamps=range(2011,2015))
    return in_circulation.mean()

@mc.stochastic(trace=True, observed=True) #stupid observed flag! got to get that right!
def circulation(entry_rate=entry_rate, loss_rate=loss_rate, value=1):
    
    mapfunc = lambda x: get_population(x, 1*entry_rate, 1*loss_rate)
    population = models['model'].apply(mapfunc) 
    
    #transform to log probability and then normalize (in the log domain, just by subtraction)
    log_distribution = np.log(population) - np.log(population.sum())
    
    #calculate the probability of the data from the distribution
    log_prob = (models['Philadelphia Samples'] * log_distribution).sum()
    
    return log_prob

mcmc = mc.MCMC(mc.Model([entry_rate, loss_rate, circulation]))

#mcmc.sample(20000)
mcmc.sample(10)

plt.hist(mcmc.trace('loss_rate')[10000:], bins=60, histtype='stepfilled', normed=True)
plt.xlabel('Loss Rate')
plt.title('Loss Rate Likelihood');

plt.hist(mcmc.trace('entry_rate')[10000:], bins=60, histtype='stepfilled', normed=True)
plt.xlabel('Entry Rate')
plt.title('Entry Rate Likelihood');

plt.hexbin(mcmc.trace('loss_rate')[:], mcmc.trace('entry_rate')[:], gridsize=30)
plt.xlabel('Loss Rate')
plt.ylabel('Entry Rate');

