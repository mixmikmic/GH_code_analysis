from auxiliaries import *
import pandas as pd

folder = "C:/nes/Options_data/"
data = pd.read_hdf(folder + 'data_ivs.h5')
data.head(2)

# for logs-models. Takes about 20 minutes
compute_all_integs(data, get_grid_basis(3, 3), 
                   dump=True, for_logs=True, verbose=True)

# for non-logs. Takes about 10 minutes
compute_all_integs(data, get_grid_basis(3, 3), 
                   dump=True, for_logs=False, verbose=True)

