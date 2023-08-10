import numpy as np
import matplotlib.pyplot as plt
from opmd_viewer import OpenPMDTimeSeries
get_ipython().magic('matplotlib')
# or `%matplotlib notebook` for inline plots

# Replace the string below, to point to your data
ts = OpenPMDTimeSeries('./diags/hdf5/')

# Interactive GUI
ts.slider()

