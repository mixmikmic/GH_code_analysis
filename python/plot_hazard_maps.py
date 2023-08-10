get_ipython().magic('matplotlib inline')
from rmtk.plotting.hazard_outputs.plot_hazard_outputs import HazardMap

hazard_map_file = "../sample_outputs/hazard/hazard_map-poe_0.02.xml"

hazard_map = HazardMap(hazard_map_file)
marker_size = 15
log_scale = False
output_file = None
output_dpi = 300
output_fmt = "png"

hazard_map.plot(log_scale, marker_size, output_file, output_dpi, output_fmt)

