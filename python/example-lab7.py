get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from sklearn.mixture import GMM # today's workhorse!

matplotlib.rcParams.update({'font.size':18})
matplotlib.rcParams.update({'font.family':'serif'})

# the data file is in the data/ directory, as per usual!
file = ''

# OPEN the data file and look at the columns, give them names since Pandas won't 
# easily understand the file conventions used by Vizier
colnames = ['col1', 'col2', 'col3']

df = pd.read_table(file, delimiter='|', skiprows=5, header=None,
                  names=colnames)

'''
Make a scatter plot of the Color vs Magnitude Diagram (CMD) using the (B-V, V) space

set the x-axis limits to be -0.5 to 2.75
set the y-axis limits to be 20 to 9

Use alpha=0.5 to see over densities easier
'''

plt.scatter(df['colN']-df['colM'], df['colM'])

''' 
Make a scatter plot of the proper motions (RA, Dec)

Set the X- and Y-axis limits to be +/- 20

Make the figure square for easier reading

Use alpha=0.5 to see over densities easier
'''

plt.scatter(df['colN'], df['colM'])

plt.xlabel('PM$_{RA}$ (mas/yr)')
plt.ylabel('PM$_{Dec}$ (mas/yr)')

'''
Remake the PM diagram above, but get rid of faint data, 
stars with missing magnitudes, and stars with outliers in the PM
'''
x = np.where((df['mag1'] < 21) & (df['Vmag'] < 20) & 
            (df['pm'] < 30)) # add upper and lower limits for RA and Dec


# reminder: you have to use the .values method to get just the data out when indexing. Annoying, but easy
plt.figure(figsize=(5,5))
plt.scatter(df['PM1'].values[x], df['PM2'].values[x])

'''
We need to feed GMM our 2-dimensional data as 1 array. It should have dimensions (N,K), where K=2 here

Also, use the index we created above to remove bad data.

Here's 1 way to do that...
'''
data = np.array( (df['PM1'].values[x], df['PM2'].values[x]) ).T # the .T does a Transpose!
data.shape

# Try it w/ and w/o the .T to see how that works...

''' 
Step 1: Initialize the type of GMM we want

How many models to fit? 
What "type" or shape of Gaussian to use? (can be ‘spherical’, ‘tied’, ‘diag’, ‘full’)

Many other options can be set here, see the GMM doc's!
'''

# PLAY WITH THESE PARAMETERS - I have intentionally made them wrong

Nmodels = 5 # tell it how many Gaussians to fit. we probably only want 2 Gaussians for this problem

ctype = 'full' # what type of Gaussian to fit?
# you can see all 4 examples here: http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_classifier.html


gmm_model = GMM(n_components=Nmodels, covariance_type=ctype)

'''
Step 2: Fit the data with the GMM

This is straight forward if your data is shaped correctly. It simply tries to fit the K-dimensional 
data (again, K=2 here) with the specified # of Gaussians.

NOTE: you can fit any number of K dimenions of data, provided the features look like K-dimensional Gaussians
'''

gmm_model.fit(data)

'''
Step 3: Make predictions or classifications based on the fit.

Since we're just trying to classify our existing data, we are just going to "predict" based 
on the input data again. 

If we had *new* data we wanted to classify, we could drop it in here instead.
'''

clabel = gmm_model.predict(data)

# there is a "label" (a prediction) for every data point we just fed it
clabel.shape

'''
Step 3b: Look at Probabilities

One of the best parts of GMM is that it produces a probability for every object being a member of
every Gaussian cluster! Thus, you could pick out only targets with very high probabilities.

'''

probs = gmm_model.predict_proba(data)
print(probs.shape) # this has dimensions (N data points, Nmodels)


# Let's look at the 100th star in our dataset.
i = 100

# does it belong to cluster 0 or 1? How certain is it?
print(probs[i,:], clabel[i])

'''
Step 4: View and Use these predictions

We have these labels, but maybe they are meaningless junk... the only way to know is to LOOK at the outputs.

Re-make your plot of the PM diagram, but color datapoints based on the predictions!

be sure to make it square again
also, set the x/y axis limits again, etc.
'''

# here's a freebie. You already have your "data" variable
plt.scatter(data[:,0], data[:,1],
            c=clabel) # and you can color using the 0's/1's label vector

'''
Now, re-plot the CMD for the cluster, but color by the predictions as above.
'''


'''
OR - you could just plot the likely cluster members only

NOTE: this may be 1 or 0. It's basically random depending on which cluster it assigned first.
'''
cl = np.where((clabel == 0))

plt.scatter(df['B'].values[x][cl] - df['V'].values[x][cl], df['V'].values[x][cl], alpha=0.5, lw=0, c='k')

# Probability of membership in cluster versus V-band magnitude

# Probability of membership in cluster versus radial distance from center of the field (using xpos and ypos)

# Comparison of your GMM cluster probabilities with the probabilities computed in the actual 
# scientific research paper, Yadav et al. (2008), using the Pmb column in the data

