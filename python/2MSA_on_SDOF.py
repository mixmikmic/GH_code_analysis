import numpy
from rmtk.vulnerability.common import utils
from rmtk.vulnerability.derivation_fragility.NLTHA_on_SDOF import MSA_utils
from rmtk.vulnerability.derivation_fragility.NLTHA_on_SDOF.read_pinching_parameters import read_parameters
from rmtk.vulnerability.derivation_fragility.NLTHA_on_SDOF import double_MSA_on_SDOF
get_ipython().magic('matplotlib inline')

capacity_curves_file = '/Users/chiaracasotto/GitHub/rmtk_data/2MSA/capacity_curves.csv'
sdof_hysteresis = "/Users/chiaracasotto/GitHub/rmtk_data/pinching_parameters.csv"

capacity_curves = utils.read_capacity_curves(capacity_curves_file)
capacity_curves = utils.check_SDOF_curves(capacity_curves)
utils.plot_capacity_curves(capacity_curves)
hysteresis = read_parameters(sdof_hysteresis)

gmrs_folder = '../../../../../rmtk_data/MSA_records'
number_models_in_DS = 1
no_bins = 2
no_rec_bin = 10
damping_ratio = 0.05
minT = 0.1
maxT = 2

filter_aftershocks = 'FALSE'
Mw_multiplier = 0.92
waveform_path = '../../../../../rmtk_data/2MSA/waveform.csv'

gmrs = utils.read_gmrs(gmrs_folder)
gmr_characteristics = MSA_utils.assign_Mw_Tg(waveform_path, gmrs, Mw_multiplier, 
                                                   damping_ratio, filter_aftershocks)
#utils.plot_response_spectra(gmrs,minT,maxT)

damage_model_file = "/Users/chiaracasotto/GitHub/rmtk_data/2MSA/damage_model_ISD.csv"

damage_model = utils.read_damage_model(damage_model_file)

degradation = False
record_scaled_folder = "../../../../../rmtk_data/2MSA/Scaling_factors"

msa = MSA_utils.define_2MSA_parameters(no_bins,no_rec_bin,record_scaled_folder,filter_aftershocks)
PDM, Sds, gmr_info = double_MSA_on_SDOF.calculate_fragility(
                        capacity_curves, hysteresis, msa, gmrs, gmr_characteristics,
                        damage_model, damping_ratio,degradation, number_models_in_DS)

IMT = 'Sa'
T = 0.47
#T = numpy.arange(0.4,1.91,0.01)
regression_method = 'max likelihood'
        
fragility_model = MSA_utils.calculate_fragility_model_damaged( PDM,gmrs,gmr_info,IMT,msa,damage_model,
                                                                        T,damping_ratio, regression_method)

minIML, maxIML = 0.01, 4

MSA_utils.plot_fragility_model(fragility_model,damage_model,minIML, maxIML)

output_type = "csv"
output_path = "../../../../../rmtk_data/2MSA/"
minIML, maxIML = 0.01, 4
tax = 'RC'

MSA_utils.save_mean_fragility(fragility_model,damage_model,tax,output_type,output_path,minIML, maxIML)



