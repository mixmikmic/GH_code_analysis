from rmtk.vulnerability.model_generator.SPBELA_approach import SPBELA
from rmtk.vulnerability.common import utils
get_ipython().magic('matplotlib inline')

building_model_file = "../../../../../rmtk_data/SPBELA/bare_frames.csv"
damage_model_file = "../../../../../rmtk_data/damage_model_spbela.csv"

no_assets = 200

building_class_model = SPBELA.read_building_class_model(building_model_file)
assets = SPBELA.generate_assets(building_class_model, no_assets)
damage_model = utils.read_damage_model(damage_model_file)
capacity_curves = SPBELA.generate_capacity_curves(assets, damage_model)

utils.plot_capacity_curves(capacity_curves)

gamma = 1.2
yielding_point_index = 1.0

capacity_curves = utils.add_information(capacity_curves, "gamma", "value", gamma)
capacity_curves = utils.add_information(capacity_curves, "yielding point", "point", yielding_point_index)

output_file = "../../../../../rmtk_data/capacity_curves_spbela.csv"

utils.save_SdSa_capacity_curves(capacity_curves, output_file)

