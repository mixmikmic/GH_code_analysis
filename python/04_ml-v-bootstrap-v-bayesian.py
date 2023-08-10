get_ipython().magic('matplotlib inline')
import pytc

# Define functions to do each fit

def fit_with_ml():
    #--------------------------------------------------------------------
    # Create a global fitting instance
    g = pytc.GlobalFit()

    # --------------------------------------------------------------------
    # Load in an experimental data set with a single-site binding model.  Ignore the first two shots
    a = pytc.ITCExperiment("ca-edta/tris-01.DH",pytc.indiv_models.SingleSite,shot_start=2)

    # Add the experiment to the fitter
    g.add_experiment(a)

    # --------------------------------------------------------------------
    # Fit the data
    g.fit()
    
    return g

def fit_with_bootstrap(num_bootstrap=1000):
    # --------------------------------------------------------------------
    # Create a global fitting instance
    g = pytc.GlobalFit()

    # --------------------------------------------------------------------
    # Load in an experimental data set with a single-site binding model.  Ignore the first two shots
    a = pytc.ITCExperiment("ca-edta/tris-01.DH",pytc.indiv_models.SingleSite,shot_start=2)

    # Add the experiment to the fitter
    g.add_experiment(a)

    # --------------------------------------------------------------------
    # Fit the data

    F = pytc.fitters.BootstrapFitter(num_bootstrap=num_bootstrap)
    g.fit(F)

    return g
    
def fit_with_bayes(num_steps=1000):
    # --------------------------------------------------------------------
    # Create a global fitting instance
    g = pytc.GlobalFit()

    # --------------------------------------------------------------------
    # Load in an experimental data set with a single-site binding model.  Ignore the first two shots
    a = pytc.ITCExperiment("ca-edta/tris-01.DH",pytc.indiv_models.SingleSite,shot_start=2)

    # Add the experiment to the fitter
    g.add_experiment(a)

    # --------------------------------------------------------------------
    # Fit the data
    F = pytc.fitters.BayesianFitter(num_steps=num_steps)
    g.fit(F)
    
    return g

print("*** ML fit ***")
ml = fit_with_ml()

print("*** Bootstrap fit ***")
bootstrap = fit_with_bootstrap(num_bootstrap=1000)

print("*** Bayesian fit ***")
bayes = fit_with_bayes(num_steps=1000)

print(ml.fit_as_csv)
fig, ax = ml.plot()
c = ml.corner_plot()

print(bootstrap.fit_as_csv)
fig, ax = bootstrap.plot()
c = bootstrap.corner_plot()

print(bayes.fit_as_csv)
fig, ax = bayes.plot()
c = bayes.corner_plot()

print("*** ML fit ***")
get_ipython().magic('timeit fit_with_ml()')
print("")

print("*** Bootstrap fit ***")
get_ipython().magic('timeit fit_with_bootstrap(1000)')
print("")

print("*** Bayesian fit ***")
get_ipython().magic('timeit fit_with_bayes(1000)')
print("")



