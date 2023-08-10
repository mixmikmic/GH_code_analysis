get_ipython().magic('matplotlib inline')
import rmtk.plotting.risk_maps.plot_risk_maps as plotrm

loss_map_file = '../sample_outputs/scenario_risk/loss-maps-structural.xml'
exposure_model = '../input_models/exposure_model_nepal.xml'

plotting_type = 2
bounding_box = 0
marker_size = 5
log_scale = True
export_map_to_csv = False

plotrm.build_map(plotting_type,loss_map_file,bounding_box,log_scale,exposure_model,marker_size,export_map_to_csv)



