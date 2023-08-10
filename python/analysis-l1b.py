from dedop.ui.analysis import inspect_l1b
get_ipython().magic('matplotlib inline')

insp = inspect_l1b("D:/EOData/DeDop/L1B.nc")

insp.lon_range

insp.lat_range

insp.plot.locations()

insp.waveform_range

insp.waveform

insp.plot.waveform_im(vmin=0, vmax=7e6)

insp.plot.waveform_hist(vmin=0, vmax=1e7, log=True)

insp.plot.waveform(ind=500)

insp.plot.waveform()

insp.plot.waveform(ref_ind=520)

insp.plot.line()

insp.plot.line(x='lon_l1b_echo_sar_ku', y='lat_l1b_echo_sar_ku')

insp.plot.im()

insp.plot.im_line()

insp.dim_names

insp.var_names

insp.dim_name_to_size

insp.dim_names_to_var_names

var = insp.dataset['scale_factor_ku_l1b_echo_plrm']

var.name, var.dtype, var.dimensions, var.shape

insp.dataset



