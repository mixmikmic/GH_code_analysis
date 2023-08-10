from rmtk.vulnerability.common import utils
from rmtk.vulnerability.derivation_fragility.hybrid_methods.N2 import N2Method

get_ipython().magic('matplotlib inline')
files_folder = '../../../../rmtk_data/portfolio_Italy'
models = utils.read_set_models(files_folder)

gmrs_folder = '../../../../rmtk_data/accelerograms'
gmrs = utils.read_gmrs(gmrs_folder)
minT = 0.1
maxT = 2
utils.plot_response_spectra(gmrs,minT,maxT)

damage_model = utils.read_damage_model('../../../../../rmtk_data/damage_model.csv')
damping = 0.05
T = 2.0

for imodel in range(len(models['name'])):
    print models['name'][imodel]
    capacity_curves = utils.read_capacity_curves(models['location'][imodel])
    utils.plot_capacity_curves(capacity_curves)    
    PDM, Sds = N2Method.calculate_fragility(capacity_curves,gmrs,damage_model,damping)
    fragility_model = utils.calculate_mean_fragility(gmrs,PDM,T,damping,'Sa',damage_model)
    utils.plot_fragility_model(fragility_model,0.01,2)



