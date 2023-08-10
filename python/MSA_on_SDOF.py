import numpy as np
from rmtk.vulnerability.common import utils
from rmtk.vulnerability.derivation_fragility.NLTHA_on_SDOF import MSA_on_SDOF
from rmtk.vulnerability.derivation_fragility.NLTHA_on_SDOF import MSA_utils
from rmtk.vulnerability.derivation_fragility.NLTHA_on_SDOF.read_pinching_parameters import read_parameters
get_ipython().magic('matplotlib inline')

capacity_curves_file = '/Users/chiaracasotto/GitHub/rmtk_data/capacity_curves_sdof_first_mode.csv'
sdof_hysteresis = "/Users/chiaracasotto/GitHub/rmtk_data/pinching_parameters.csv"

capacity_curves = utils.read_capacity_curves(capacity_curves_file)
capacity_curves = utils.check_SDOF_curves(capacity_curves)
utils.plot_capacity_curves(capacity_curves)
hysteresis = read_parameters(sdof_hysteresis)

gmrs_folder = "../../../../../rmtk_data/MSA_records"
minT, maxT = 0.1, 2.0
no_bins = 2
no_rec_bin = 10
record_scaled_folder = "../../../../../rmtk_data/Scaling_factors"

gmrs = utils.read_gmrs(gmrs_folder)
#utils.plot_response_spectra(gmrs, minT, maxT)

damage_model_file = "../../../../../rmtk_data/damage_model_Sd.csv"

damage_model = utils.read_damage_model(damage_model_file)

damping_ratio = 0.05
degradation = False

msa = {}; msa['n. bins']=no_bins; msa['records per bin']=no_rec_bin; msa['input folder']=record_scaled_folder
PDM, Sds, IML_info = MSA_on_SDOF.calculate_fragility(capacity_curves, hysteresis, msa, gmrs, 
                                                      damage_model, damping_ratio, degradation)

IMT = "Sa"
T = 0.47
#T = np.arange(0.4,1.91,0.01)
regression_method = "least squares"

fragility_model = MSA_utils.calculate_fragility_model(PDM,gmrs,IML_info,IMT,msa,damage_model,
                                                                        T,damping_ratio, regression_method)

minIML, maxIML = 0.01, 4
utils.plot_fragility_model(fragility_model, minIML, maxIML)

print fragility_model['damage_states'][0:]

taxonomy = "HI_Intact_v4_lq"
minIML, maxIML = 0.01, 3.00
output_type = "csv"
output_path = "../../../../../phd_thesis/results/damping_0.39/"

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



