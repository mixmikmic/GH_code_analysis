get_ipython().magic('matplotlib inline')
from rmtk.plotting.collapse_maps import plot_collapse_maps as plt

collapse_map = '../input_models/collapse_map.xml'
exposure_model = '../input_models/exposure_model.xml'

plotting_type = 2
bounding_box = 0
marker_size = 10
log_scale = True
export_map_to_csv = True

plt.build_map(plotting_type,collapse_map,bounding_box,log_scale,exposure_model,marker_size,export_map_to_csv)



