get_ipython().magic('matplotlib inline')
import pytc

# --------------------------------------------------------------------
# Create a global fitting instance
g = pytc.GlobalFit()

# --------------------------------------------------------------------
# Load in an experimental data set with a single site competitor model,
# with fake 10 uM competitor in the cell and syringe.
a = pytc.ITCExperiment("ca-edta/tris-01.DH",pytc.indiv_models.SingleSiteCompetitor,
                       C_cell=1e-5,C_syringe=1e-5,shot_start=2)
# Add the experiment to the fitter
g.add_experiment(a)

# --------------------------------------------------------------------
# Fit the data
g.fit()

# --------------------------------------------------------------------
# Show the results
fig, ax = g.plot()
print(g.fit_as_csv)
c = g.corner_plot()



