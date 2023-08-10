import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Each CSV File is organized like this
df = pd.read_csv('data/House_2.csv', index_col=0, parse_dates=True)
df.head()

# Prepare to visualize usage of different applicances in different hours of a day
by_hours_df = df.iloc[:, 2:].groupby(df.index.hour).mean()
# Rename the applicances according to the metadata provided, for more readability
by_hours_df = by_hours_df.rename(index=str, columns={
    "Appliance1":"Fridge-Freezer",
    "Appliance2":"Washing Machine",
    "Appliance3":"Dishwasher",
    "Appliance4":"Television Site",
    "Appliance5":"Microwave",
    "Appliance6":"Toaster",
    "Appliance7":"Hi-Fi",
    "Appliance8":"Kettle",
    "Appliance9":"Overhead Fan"
})
# The data frame aggregated by hours of a day looks like:
by_hours_df

# Set size of plot
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
# Stackplot
ax.stackplot(list(range(24)), np.transpose(by_hours_df.values), labels=by_hours_df.columns.values)
plt.xlabel('Hour in a day', fontsize=18)
plt.ylabel('Watts averaged per hour', fontsize=18)
plt.xticks(np.arange(0, 24, 1.0))
ax.legend(loc=2)

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)

plt.subplot(241)
# Usage of various appliances
plt.plot(list(range(24)), by_hours_df['Fridge-Freezer'].values)
plt.title('Fridge Freezer')

plt.subplot(242)
# Usage of various appliances
plt.plot(list(range(24)), by_hours_df['Washing Machine'].values)
plt.title('Washing Machine')

plt.subplot(243)
# Usage of various appliances
plt.plot(list(range(24)), by_hours_df['Dishwasher'].values)
plt.title('Dishwasher')

plt.subplot(244)
# Usage of various appliances
plt.plot(list(range(24)), by_hours_df['Television Site'].values)
plt.title('Television Site')

plt.subplot(245)
# Usage of various appliances
plt.plot(list(range(24)), by_hours_df['Microwave'].values)
plt.title('Microwave')

plt.subplot(246)
# Usage of various appliances
plt.plot(list(range(24)), by_hours_df['Toaster'].values)
plt.title('Toaster')

plt.subplot(247)
# Usage of various appliances
plt.plot(list(range(24)), by_hours_df['Hi-Fi'].values)
plt.title('Hi-Fi')

plt.subplot(248)
# Usage of various appliances
plt.plot(list(range(24)), by_hours_df['Kettle'].values)
plt.title('Kettle')

