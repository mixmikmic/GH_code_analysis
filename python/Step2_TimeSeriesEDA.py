import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("../data/device_failure.csv")
df.columns = ['date', 'device', 'failure', 'a1', 'a2','a3','a4','a5','a6','a7','a8','a9']
fcols = ['a1', 'a2','a3','a4','a5','a6','a7','a8','a9']
df.loc[:,'date'] = pd.to_datetime(df['date'])

failed_devs = pd.DataFrame(df[df['failure'] == 1].device.unique())
failed_devs.columns = ["device"]
failed_devs_hist = pd.merge(df, failed_devs, on=["device"])

good_devs = pd.DataFrame(list(set(df.device.unique()) - set(failed_devs["device"])))
good_devs.columns = ["device"]
good_devs_hist = pd.merge(df, good_devs, on=["device"])

def plot_history(tdf, feature, devname):
    fdev = tdf[tdf["device"] == devname]
    fdev.set_index("date", inplace=True)
    fdev[feature].plot()

def plot_sample_history(tdf, dev_list_df, sample_cnt, feature):
    #Get a sample of devices and their history
    sample_dev_df = dev_list_df.sample(sample_cnt)
    sample_dev_hist = pd.merge(tdf, sample_dev_df, on=["device"])
    for device in sample_dev_df["device"]:
        fig, axs = plt.subplots(1)
        fig.set_size_inches(6,2)
        plot_history(sample_dev_hist, feature, device)

plot_sample_history(failed_devs_hist, failed_devs, 3, "a2")

plot_sample_history(good_devs_hist, good_devs, 3, "a2")

plot_sample_history(failed_devs_hist, failed_devs, 3, "a1")

plot_sample_history(good_devs_hist, good_devs, 3, "a1")

df[df["a7"] != df["a8"]]



