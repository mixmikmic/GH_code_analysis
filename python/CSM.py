from rmtk.vulnerability.derivation_fragility.hybrid_methods.CSM import capacitySpectrumMethod
from rmtk.vulnerability.common import utils
get_ipython().magic('matplotlib inline')

capacity_curves_file = "../../../../../../rmtk_data/capacity_curves_Sa-Sd.csv"

capacity_curves = utils.read_capacity_curves(capacity_curves_file)
utils.plot_capacity_curves(capacity_curves)

gmrs_folder = "../../../../../../rmtk_data/accelerograms"
minT, maxT = 0.1, 2.0

gmrs = utils.read_gmrs(gmrs_folder)
#utils.plot_response_spectra(gmrs, minT, maxT)

damage_model_file = "../../../../../../rmtk_data/damage_model.csv"

damage_model = utils.read_damage_model(damage_model_file)

damping_model = "Iwan_1980"
damping_ratio = 0.05
PDM, Sds = capacitySpectrumMethod.calculate_fragility(capacity_curves, gmrs, damage_model, 
                                                      damping_model, damping_ratio)

IMT = "Sa"
period = 0.3
regression_method = "least squares"

fragility_model = utils.calculate_mean_fragility(gmrs, PDM, period, damping_ratio, 
                                                 IMT, damage_model, regression_method)

minIML, maxIML = 0.01, 2.00

utils.plot_fragility_model(fragility_model, minIML, maxIML)
# utils.plot_fragility_stats(fragility_statistics,minIML,maxIML)

taxonomy = "RC"
minIML, maxIML = 0.01, 2.00
output_type = "csv"
output_path = "../../../../../../rmtk_data/output/"

utils.save_mean_fragility(taxonomy, fragility_model, minIML, maxIML, output_type, output_path)

cons_model_file = "../../../../../../rmtk_data/cons_model.csv"
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
output_path = "../../../../../../rmtk_data/output/"

utils.save_vulnerability(taxonomy, vulnerability_model, output_type, output_path)

