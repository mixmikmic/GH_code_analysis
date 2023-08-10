# Standard modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from getpass import getuser

# Custom modules
from closest import *
from jpm_time_conversions import sod_to_hhmmss

# TODO: These will be a part of the function definition
number_of_samples_to_average = 6
save_path = '/Users/' + getuser() + '/Dropbox/Research/Postdoc_NASA/Analysis/Coronal Dimming Analysis/EVE Precision/'
reload_eve_data = False
verbose = False

# Get data for a quiet period - beginning of the below day is very quiet in 171 Å
if reload_eve_data == True:
    # TODO: Asking Don Woodraska if there is an easy way to get the EVE lines from sunpy
    print('Fetching remote EVE data')
else:
    from scipy.io.idl import readsav
    eveLines = readsav(save_path + 'EVE Line Data.sav')

end_index = closest(eveLines['sod'], 3600)
timestamp_iso = '2013-01-28 ' + sod_to_hhmmss(eveLines['sod'])[:end_index]
eve_lines = eveLines['evelines'][:end_index, :]

selected_lines_dictionary = {'94': pd.Series(eve_lines[:, 0], index=timestamp_iso),
                             '132': pd.Series(eve_lines[:, 2], index=timestamp_iso),
                             '171': pd.Series(eve_lines[:, 3], index=timestamp_iso),
                             '177': pd.Series(eve_lines[:, 4], index=timestamp_iso),
                             '180': pd.Series(eve_lines[:, 5], index=timestamp_iso),
                             '195': pd.Series(eve_lines[:, 6], index=timestamp_iso),
                             '202': pd.Series(eve_lines[:, 7], index=timestamp_iso),
                             '211': pd.Series(eve_lines[:, 8], index=timestamp_iso),
                             '284': pd.Series(eve_lines[:, 10], index=timestamp_iso),
                             '335': pd.Series(eve_lines[:, 12], index=timestamp_iso),}
selected_lines = pd.DataFrame(selected_lines_dictionary)
selected_lines.index.name = 'Timestamp'
selected_lines.head() # TODO: Remove this line

# Compute normalized precision time series
group_to_average = selected_lines.groupby(np.arange(len(selected_lines)) // number_of_samples_to_average)
precision_time_series = group_to_average.std() / (group_to_average.mean() * np.sqrt(number_of_samples_to_average))

# Take average of normalized precision time series over the hour long period
precision = precision_time_series.mean()

if verbose:
    print(precision)
    
# return precision # TODO: Uncomment this for function

precision # TODO: Remove this line for function

