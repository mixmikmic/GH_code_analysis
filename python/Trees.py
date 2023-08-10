# If you want the figures to appear in the notebook, use
# %matplotlib notebook

# If you want the figures to appear in separate windows, use
# %matplotlib qt

# To switch from one to another, you have to select Kernel->Restart

get_ipython().magic('matplotlib inline')

from modsim import *

years = [2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30,
         35, 40, 45, 50, 55, 60, 65, 70]

site65 = Series([1.4, 1.56, 2.01, 2.76, 3.79, 6.64, 10.44, 
                 23.26, 37.65, 51.66, 65.00, 77.50, 89.07, 
                 99.66, 109.28, 117.96, 125.74, 132.68, 138.84],
               index=years)

site45 = Series([1.4, 1.49, 1.75, 2.18, 2.78, 4.45, 6.74,
                 14.86, 25.39, 35.60, 45.00, 53.65, 61.60,
                68.92, 75.66, 81.85, 87.56, 92.8, 97.63],
               index=years)

site = site65

plot(site)

def update(year, mass, state):
    height = mass**(1/tree.dimension)
    area = height**2
    growth = state.alpha * area * (1 - height/tree.K)
    return mass + growth

t0 = years[0]
h0 = site[t0]

tree = State(mass=1, alpha=2.1, dimension=2.55, K=155)

m0 = h0**tree.dimension

tree.masses = Series({t0: m0})

for i in range(t0, 70):
    tree.masses[i+1] = update(i, tree.masses[i], tree)

# TODO: check whether there are any labeled lines before calling lengend,
# or suppress the warning

heights = tree.masses**(1.0/tree.dimension)

plot(heights, label='model')
plot(site, label='data')
decorate()

tree = State(t0=t0, h0=1.4, alpha=4, dimension=2.75, K=180)

def run_model(state):
    m0 = h0**tree.dimension
    tree.masses = Series({state.t0: m0})

    for i in range(t0, 70):
        tree.masses[i+1] = update(i, tree.masses[i], state)

run_model(tree)
heights = tree.masses**(1/tree.dimension)

def print_errors(model, data):
    abs_err = abs(model[data.index] - data)
    rel_err = abs_err / data * 100
    print(rel_err)

def error(model, data):
    abs_err = abs(model[data.index] - data)
    rel_err = abs_err / data * 100
    return abs_err.mean()

print_errors(heights, site)

error(heights, site)

from scipy.optimize import fmin

alpha = 2.1
dimension = 2.55
K = 155

x0 = [alpha, dimension, K]

def func(x, tree):
    tree.alpha, tree.dimension, tree.K = x
    run_model(tree)
    heights = tree.masses**(1/tree.dimension)
    return error(heights, site)

func(x0, tree)

args = (tree,)
params = fmin(func, x0, args=args)
params

tree.alpha, tree.dimension, tree.K = params
run_model(tree)
heights = tree.masses**(1/tree.dimension)

plot(heights, label='model')
plot(site, label='data')
decorate()

error(heights, site)



