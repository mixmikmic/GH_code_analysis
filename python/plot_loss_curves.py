get_ipython().magic('matplotlib inline')
from rmtk.plotting.loss_curves import plot_loss_curves as plotlc

loss_curves_file = '../sample_outputs/classical_risk/loss-curves-structural.xml'
assets_list = ['a1', 'a10', 'a100']

log_scale_x = True
log_scale_y = True

plotlc.plot_loss_curves(loss_curves_file, assets_list, log_scale_x, log_scale_y)

