get_ipython().magic('matplotlib inline')
from dc_interact import dc_resistivity
from ipywidgets import interact
interact(
    dc_resistivity,
    log_sigma_background=(-4,4),
    log_sigma_block=(-4,4),
    plot_type=['potential','conductivity','current']
);

