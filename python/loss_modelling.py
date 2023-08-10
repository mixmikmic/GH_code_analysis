from rmtk.risk.event_loss_tables.loss_modelling import lossModelling
get_ipython().magic('matplotlib inline')

import os
print os.getcwd()

event_loss_table_folder = 'Nepal'
total_cost = 62e9
investigationTime = 10000
return_periods = [100,475,950]

save_elt_csv = False
save_ses_csv = False

lossModelling(event_loss_table_folder,save_ses_csv,save_elt_csv,total_cost,investigationTime,return_periods)

from loss_modelling import selectRuptures 
return_periods = [500,5000]
rups_for_return_period = 10
selectRuptures(event_loss_table_folder,return_periods,rups_for_return_period)



