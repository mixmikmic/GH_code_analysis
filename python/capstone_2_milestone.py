import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

time_train_full = pd.read_csv("train_2_final.csv")
time_test = time_train_full.iloc[:, -62:].assign(Page = time_train_full.loc[:, "Page"])
time_train = time_train_full.iloc[:, -72:-62].assign(Page = time_train_full.loc[:, "Page"])
time_train.head()

time_train_median = time_train.median(axis = 1, skipna = True).fillna(0)
time_test_melt = pd.melt(time_test, id_vars = ["Page"])
time_train_median_frame = time_train.loc[:, "Page"].to_frame(name = "Page").assign(page_median = time_train_median)
time_test_melt_join = time_test_melt.merge(time_train_median_frame, on = "Page", how = "left")
page_median_log = time_test_melt_join.loc[:, "page_median"]
page_median_log[page_median_log == 0] = 1
page_median_log = np.log(page_median_log.values)
page_observed_log = time_test_melt_join.loc[:, "value"]
page_observed_log[page_observed_log == 0] = 1
page_observed_log = np.log(page_observed_log.values)
plt.scatter(page_median_log, page_observed_log)



