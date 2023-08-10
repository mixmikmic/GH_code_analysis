from SimPEG import *
from SimPEG.Examples import DC_PseudoSection_Simulation

from ipywidgets import interactive, FloatText, FloatSlider, ToggleButtons #interactive plots!

get_ipython().magic('matplotlib inline')

fig, ax = DC_PseudoSection_Simulation.run()

Simul_Fct = lambda a,b,n, sig0, sig1, sig2, stype: DC_PseudoSection_Simulation.run(param = np.r_[a,b,n], sig=np.r_[sig0,sig1,sig2], stype = stype)

interactive(Simul_Fct, a=FloatText(min=10.,max=40.,step=5.,value=30.),
                        b=FloatText(min=10.,max=40.,step=5.,value=30.),
                        n=FloatText(min=1,max=30,step=5,value=10.),
                        sig0=FloatText(min=1e-4,max=1e+4,step=1e+1,value=1e-2),
                        sig1=FloatText(min=1e-4,max=1e+4,step=1e+1,value=1e-1),
                        sig2=FloatText(min=1e-4,max=1e+4,step=1e+1,value=1e-3),
                        stype=ToggleButtons(options=['pdp','dpdp'],value='dpdp'))
                       





