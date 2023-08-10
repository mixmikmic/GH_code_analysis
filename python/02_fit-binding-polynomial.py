get_ipython().magic('matplotlib inline')
import pytc

# --------------------------------------------------------------------
# Create a global fitting instance
g = pytc.GlobalFit()

# --------------------------------------------------------------------
# Load in an experimental data set with a single-site binding polynomial model
a = pytc.ITCExperiment("ca-edta/tris-01.DH",pytc.indiv_models.BindingPolynomial,num_sites=1,shot_start=2)
# Add the experiment to the fitter
g.add_experiment(a)

# --------------------------------------------------------------------
# Fit the data
g.fit()

# --------------------------------------------------------------------
# Show the results
fig, ax = g.plot()
c = g.corner_plot()
print(g.fit_as_csv)



