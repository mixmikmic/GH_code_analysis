from rmtk.vulnerability.model_generator.point_dispersion import point_dispersion as pd
from rmtk.vulnerability.common import utils
get_ipython().magic('matplotlib inline')

Sa_means = [0.40, 0.40, 0.40, 0.40]
Sa_covs  = [0.20, 0.20, 0.20, 0.20]
Sd_means = [0.03, 0.05, 0.08, 0.1]
Sd_covs  = [0.20, 0.20, 0.20, 0.20]
distribution = "normal"
Sa_corr = 0.99999
Sd_corr = 0.99999
Sa_Sd_corr = 0.5
truncation_level = 1
no_capacity_curves = 50
capacity_curves = pd.generate_capacity_curves(Sa_means, Sa_covs, Sd_means, Sd_covs,
                                              distribution, no_capacity_curves, 
                                              Sa_corr, Sd_corr, Sa_Sd_corr, truncation_level)
utils.plot_capacity_curves(capacity_curves)

gamma = 1.2
height = 3.0
elastic_period = 0.6
yielding_point_index = 1
capacity_curves = utils.add_information(capacity_curves, 'gamma', 'value', gamma)
capacity_curves = utils.add_information(capacity_curves, 'heights', 'value', height)
capacity_curves = utils.add_information(capacity_curves, 'periods', 'calculate', 1)
capacity_curves = utils.add_information(capacity_curves, 'yielding point', 'point', yielding_point_index)

output_file = "../../../../../rmtk_data/capacity_curves_point.csv"
utils.save_SdSa_capacity_curves(capacity_curves, output_file)



