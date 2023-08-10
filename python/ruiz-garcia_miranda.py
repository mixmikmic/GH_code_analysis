from rmtk.vulnerability.derivation_fragility.R_mu_T_dispersion.ruiz_garcia_miranda import RGM2007 
from rmtk.vulnerability.common import utils
import scipy.stats as stat
get_ipython().magic('matplotlib inline')

capacity_curves_file = "../../../../../../rmtk_data/capacity_curves_Vb-droof.csv"
input_spectrum = "../../../../../../rmtk_data/FEMAP965spectrum.txt"

capacity_curves = utils.read_capacity_curves(capacity_curves_file)
utils.plot_capacity_curves(capacity_curves)
Sa_ratios = utils.get_spectral_ratios(capacity_curves, input_spectrum)

idealised_type = "bilinear"

idealised_capacity = utils.idealisation(idealised_type, capacity_curves)
utils.plot_idealised_capacity(idealised_capacity, capacity_curves, idealised_type)

damage_model_file = "../../../../../../rmtk_data/damage_model_ISD.csv"

damage_model = utils.read_damage_model(damage_model_file)

montecarlo_samples = 50

fragility_model = RGM2007.calculate_fragility(capacity_curves, idealised_capacity, damage_model, montecarlo_samples, Sa_ratios)

minIML, maxIML = 0.01, 2.00

utils.plot_fragility_model(fragility_model, minIML, maxIML)

taxonomy = "RC"
minIML, maxIML = 0.01, 2.00
output_type = "csv"
output_path = "../../../../../../rmtk_data/output/"

utils.save_mean_fragility(taxonomy, fragility_model, minIML, maxIML, output_type, output_path)

cons_model_file = "../../../../../../rmtk_data/cons_model.csv"
imls = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 
        0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00]
distribution_type = "lognormal"

cons_model = utils.read_consequence_model(cons_model_file)
vulnerability_model = utils.convert_fragility_vulnerability(fragility_model, cons_model, 
                                                            imls, distribution_type)

utils.plot_vulnerability_model(vulnerability_model)

taxonomy = "RC"
output_type = "csv"
output_path = "../../../../../../rmtk_data/output/"

utils.save_vulnerability(taxonomy, vulnerability_model, output_type, output_path)



