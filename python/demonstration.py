import libOPAL.DSE 
libOPAL.DSE.run('./example_MLP.cfg')

get_ipython().magic('matplotlib notebook')
from libOPAL.results import ResultsBrowser

results_plot_MLP = ResultsBrowser('./results/example_MLP')

results_plot_CNN = ResultsBrowser('./results/example_CNN')



