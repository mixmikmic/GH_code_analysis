from dedop.ui.inspect import inspect_l1b_product
get_ipython().magic('matplotlib inline')

insp = inspect_l1b_product(__L1B_FILE_PATH__)

insp.lon_range

insp.lat_range

insp.plot.locations()

insp.waveform_range

insp.waveform

insp.plot.waveform_im(vmin=0, vmax=7e6)

insp.plot.waveform_hist(vmin=0, vmax=1e7, log=True)

insp.plot.waveform_line(ind=500)

insp.plot.waveform_line()

insp.plot.waveform_line(ref_ind=520)

insp.plot.waveform_3d_surf()

insp.plot.line()

insp.plot.line(x='lon_l1b_echo_sar_ku', y='lat_l1b_echo_sar_ku')

insp.plot.im()

insp.plot.im_line()

insp.dim_names

insp.var_names

insp.dim_name_to_size

insp.dim_names_to_var_names

var = insp.dataset['i2q2_meas_ku_l1b_echo_sar_ku']

var.name, var.dtype, var.dimensions, var.shape

insp.dataset



