# If you want the figures to appear in the notebook, 
# and you want to interact with them, use
# %matplotlib notebook

# If you want the figures to appear in the notebook, 
# and you don't want to interact with them, use
# %matplotlib inline

# If you want the figures to appear in separate windows, use
# %matplotlib qt5

# To switch from one to another, you have to select Kernel->Restart

get_ipython().run_line_magic('matplotlib', 'inline')

from modsim import *

from pandas import read_excel

filename = 'data/China_Children_Population.xlsx'
table = read_excel(filename, header=0, index_col=0, decimal='M')
table

table.head()

table.tail()

table.columns = ['china']
china = table.china

china.values

china.index

def plot_estimates(table):
    """Plot world population estimates.
    
    table: DataFrame with columns 'un' and 'census'
    """
    china = table.china / 1000
    
    plot(china, ':', color='darkblue', label='Real data')
    
    decorate(xlabel='Time (Year)',
             xlim=[1960, 2015],
             ylabel='Child Population of China (Million)')

newfig()
plot_estimates(table)

def update_func(pop, t, system):
    """Compute the population next year.
    
    pop: current population
    t: current year
    system: system object containing parameters of the model
    
    returns: population next year
    """
    if t < 1975:
        net_growth = system.alpha1 * pop
    elif t < 1985:
        net_growth = system.alpha2 * pop
    elif t < 1995:
        net_growth = system.alpha3 * pop
    elif t < 2000:
        net_growth = system.alpha4 * pop
    elif t < 2005:
        net_growth = system.alpha5 * pop
    elif t < 2010:
        net_growth = system.alpha6 * pop
    else:
        net_growth = system.alpha7 * pop
        
    return pop + net_growth

def run_simulation(system, update_func):
    """Simulate the system using any update function.
    
    Adds TimeSeries to `system` as `results`.

    system: System object
    update_func: function that computes the population next year
    """
    results = TimeSeries()
    results[system.t0] = system.p0
    for t in linrange(system.t0, system.t_end):
        results[t+1] = update_func(results[t], t, system)
    system.results = results
    
system = System(t0=1960,
                t_end=2015,
                p0=china[1960] / 1000,
                alpha1 = 0.024,
               alpha2 = -0.012, 
               alpha3 = 0.003,
               alpha4 = -0.012,
               alpha5 = -0.035,
               alpha6 = -0.019,
               alpha7 = 0.005)

run_simulation(system, update_func)
newfig()
plot(system.results, '--', color='gray', label='Model')
plot_estimates(table)
savefig('model.png', dpi=300, bbox_inches='tight')

