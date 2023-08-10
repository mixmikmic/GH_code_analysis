get_ipython().magic('matplotlib inline')
import pytc

# --------------------------------------------------------------------
# Create a global fitting instance
g = pytc.GlobalFit()

# --------------------------------------------------------------------
# Load in an experimental data set with a single-site binding model.  Ignore the first two shots
a = pytc.ITCExperiment("ca-edta/tris-01.DH",pytc.indiv_models.SingleSite,shot_start=2)

# Add the experiment to the fitter
g.add_experiment(a)

# Create the Bayesian fitter

# With these parameters, we start near the maximum likelihood solution and explore around it
F = pytc.fitters.BayesianFitter(num_steps=500,ml_guess=True,initial_walker_spread=0.0001,burn_in=0.10)

# --------------------------------------------------------------------
# Fit the data
g.fit(F)

# --------------------------------------------------------------------
# Show the results
fig, ax = g.plot()
c = g.corner_plot()
print(g.fit_as_csv)

# --------------------------------------------------------------------
# Create a global fitting instance
g = pytc.GlobalFit()

# --------------------------------------------------------------------
# Load in an experimental data set with a single-site binding model.  Ignore the first two shots
a = pytc.ITCExperiment("ca-edta/tris-01.DH",pytc.indiv_models.SingleSite,shot_start=2)

# Add the experiment to the fitter
g.add_experiment(a)

# Create the Bayesian fitter

# More agnostic fitter.  Make broad spread of initial walker positions, and run for a long time
F = pytc.fitters.BayesianFitter(num_steps=3000,ml_guess=True,initial_walker_spread=0.1,burn_in=0.25)

# --------------------------------------------------------------------
# Fit the data
g.fit(F)

# --------------------------------------------------------------------
# Show the results
fig, ax = g.plot()
c = g.corner_plot()
print(g.fit_as_csv)




