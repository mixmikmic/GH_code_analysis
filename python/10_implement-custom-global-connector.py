get_ipython().magic('matplotlib inline')
import pytc

# --------------------------------------------------------------------
# define buffer ionization enthalpies.
# goldberg et al (2002) Journal of Physical and Chemical Reference Data 31 231,  doi: 10.1063/1.1416902
TRIS_IONIZATION_DH = 47.45/4.184*1000 # cal/mol
IMID_IONIZATION_DH = 36.64/4.184*1000 # cal/mol

class MyNumProtons(pytc.GlobalConnector):
    param_guesses = {"num_H":0.1,"dH_intrinsic":0.0}
    required_data = ["ionization_enthalpy"]
    
    def dH(self,experiment):

        return self.dH_intrinsic + self.num_H*experiment.ionization_enthalpy


# --------------------------------------------------------------------
# Create a global fitting instance
g = pytc.GlobalFit()
num_protons = MyNumProtons("np")

# ------------------------------------------------------------------------------------
# Tris buffer experiment

tris = pytc.ITCExperiment("ca-edta/tris-01.DH",pytc.indiv_models.SingleSite,shot_start=2)
tris.ionization_enthalpy = TRIS_IONIZATION_DH

g.add_experiment(tris)
g.link_to_global(tris,"dH",num_protons.dH)
g.link_to_global(tris,"K","K_global")

# ------------------------------------------------------------------------------------
# Imidazole buffer experiment

imid = pytc.ITCExperiment("ca-edta/imid-01.DH",pytc.indiv_models.SingleSite,shot_start=2)
imid.ionization_enthalpy = IMID_IONIZATION_DH

g.add_experiment(imid)
g.link_to_global(imid,"dH",num_protons.dH)
g.link_to_global(imid,"K","K_global")

# --------------------------------------------------------------------
# Do a global fit
g.fit()

# Show the results
fig, ax = g.plot()
c = g.corner_plot()
print(g.fit_as_csv)



