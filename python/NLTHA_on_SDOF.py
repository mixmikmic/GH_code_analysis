from rmtk.vulnerability.common import utils
from rmtk.vulnerability.derivation_fragility.NLTHA_on_SDOF import NLTHA_on_SDOF
from rmtk.vulnerability.derivation_fragility.NLTHA_on_SDOF.read_pinching_parameters import read_parameters
get_ipython().magic('matplotlib inline')

capacity_curves_file = "../../../../../rmtk_data/capacity_curves_point.csv"
sdof_hysteresis = "Default"
#sdof_hysteresis = "../../../../../rmtk_data/pinching_parameters.csv"

capacity_curves = utils.read_capacity_curves(capacity_curves_file)
capacity_curves = utils.check_SDOF_curves(capacity_curves)
utils.plot_capacity_curves(capacity_curves)
hysteresis = read_parameters(sdof_hysteresis)

gmrs_folder = '../../../../../rmtk_data/accelerograms'
minT, maxT = 0.0, 2.0
gmrs = utils.read_gmrs(gmrs_folder)
#utils.plot_response_spectra(gmrs, minT, maxT)

damage_model_file = "../../../../../rmtk_data/damage_model.csv"
damage_model = utils.read_damage_model(damage_model_file)

damping_ratio = 0.05
degradation = True
PDM, Sds = NLTHA_on_SDOF.calculate_fragility(capacity_curves, hysteresis, gmrs, damage_model, damping_ratio, degradation)
utils.save_result(PDM,'../../../../../rmtk_data/PDM.csv')

minT, maxT,stepT  = 0.0, 2.0, 0.1
regression_method = 'least squares'
utils.evaluate_optimal_IM(gmrs,PDM,minT,maxT,stepT,damage_model,damping_ratio,regression_method)

IMT = "Sa"
T = 0.7
utils.export_IMLs_PDM(gmrs,T,PDM,damping_ratio,damage_model,'../../../../../rmtk_data/IMLs_PDM.csv')
fragility_model = utils.calculate_mean_fragility(gmrs, PDM, T, damping_ratio,IMT, damage_model, regression_method)

minIML, maxIML = 0.0, 1.2
utils.plot_fragility_model(fragility_model, minIML, maxIML)
utils.plot_fragility_scatter(fragility_model, minIML, maxIML, PDM, gmrs, IMT, T, damping_ratio)

taxonomy = "RC"
minIML, maxIML = 0.01, 2.00
output_type = "csv"
output_path = "../../../../../rmtk_data/output/"
utils.save_mean_fragility(taxonomy, fragility_model, minIML, maxIML, output_type, output_path)

cons_model_file = "../../../../../rmtk_data/cons_model.csv"
imls = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 
        0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 
        2.20, 2.40, 2.60, 2.80, 3.00, 3.20, 3.40, 3.60, 3.80, 4.00]
distribution_type = "lognormal"
cons_model = utils.read_consequence_model(cons_model_file)
vulnerability_model = utils.convert_fragility_vulnerability(fragility_model, cons_model, 
                                                            imls, distribution_type)

utils.plot_vulnerability_model(vulnerability_model)

taxonomy = "RC"
output_type = "csv"
output_path = "../../../../../rmtk_data/output/"
utils.save_vulnerability(taxonomy, vulnerability_model, output_type, output_path)

