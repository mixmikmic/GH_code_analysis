import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
# Read the data and plot a histogram
df = pd.read_csv('20170419_data_bootstrap.csv', header=None)
df.hist()
plt.title('Data from Unknown Distribution')
plt.xlabel('Value');

import numpy as np
# random.choice takes a data array, and you provide it with 
# the size of samples that you want, with replacement or
# without replacement
boot = np.random.choice(df[0].values, 100, replace=True)
# Let's look at the histogram of the data again, to compare
# it to the first figure.
boot_df = pd.DataFrame(boot)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
plt.suptitle('Sampling with Replacement, 100 Samples')
ax1.hist(boot_df[0].values, normed=1)
ax1.set_xlabel('Bootstrapped Data')
ax2.hist(df[0].values, normed=1)
ax2.set_xlabel('Original Data');

import numpy as np
# random.choice takes a data array, and you provide it with 
# the size of samples that you want, with replacement or
# without replacement
boot2 = np.random.choice(df[0].values, 10000, replace=True)
# Let's look at the histogram of the data again, to compare
# it to the first figure.
boot_df2 = pd.DataFrame(boot2)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
plt.suptitle('Sampling with Replacement, 10000 Samples')
ax1.hist(boot_df2[0].values, normed=1)
ax1.set_xlabel('Bootstrapped Data')
ax2.hist(df[0].values, normed=1)
ax2.set_xlabel('Original Data');

num_hist = int(1e6)
# Run the Monte Carlo simulation with the above number of histories
mc_boot = []
for _ in range(num_hist):
    # Random sampling with replacement
    boot3 = np.random.choice(df[0].values, 10000, replace=True)
    # Save the mean of the random sampling in the mc_boot array
    mc_boot.append(np.mean(boot3))
# Plot the results
mc_df = pd.DataFrame(mc_boot)
mc_df.hist();
plt.title('Distribution of the Sampling Means')
plt.xlabel('Values');

