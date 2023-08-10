get_ipython().magic('matplotlib inline')
from rmtk.vulnerability.common import utils
from rmtk.vulnerability.mdof_to_sdof.first_mode import first_mode

pushover_file = "../../../../../rmtk_data/capacity_curves_Vb-dfloor.csv"
idealised_type = 'quadrilinear'; # 'bilinear', 'quadrilinear' or 'none'

capacity_curves = utils.read_capacity_curves(pushover_file)
[sdof_capacity_curves, sdof_idealised_capacity] = first_mode.mdof_to_sdof(capacity_curves, idealised_type)

capacity_to_save = sdof_idealised_capacity

utils.save_SdSa_capacity_curves(capacity_to_save,'../../../../../rmtk_data/capacity_curves_sdof_first_mode.csv')

if idealised_type is not 'none':
    idealised_capacity = utils.idealisation(idealised_type, sdof_capacity_curves)
    utils.plot_idealised_capacity(idealised_capacity, sdof_capacity_curves, idealised_type)
else:
    utils.plot_capacity_curves(capacity_curves)
    utils.plot_capacity_curves(sdof_capacity_curves)

deformed_shape_file =  "../../../../../rmtk_data/ISD_Sd.csv"

[ISD_vectors, Sd_vectors] = first_mode.define_deformed_shape(capacity_curves, deformed_shape_file)



