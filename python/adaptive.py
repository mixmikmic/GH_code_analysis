get_ipython().magic('matplotlib inline')
from rmtk.vulnerability.common import utils
from rmtk.vulnerability.mdof_to_sdof.adaptive import adaptive

pushover_file = "../../../../../rmtk_data/capacity_curves_Vb-dfloor_adaptive.csv"

capacity_curves = utils.read_capacity_curves(pushover_file)
sdof_capacity_curves = adaptive.mdof_to_sdof(capacity_curves)
utils.save_SdSa_capacity_curves(sdof_capacity_curves,'../../../../../rmtk_data/capacity_curves_sdof_adaptive.csv')

utils.plot_capacity_curves(capacity_curves)
utils.plot_capacity_curves(sdof_capacity_curves)

