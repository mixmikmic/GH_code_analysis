import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")

# load the data
nfl_trends = pd.read_csv("data/fivethirtyeight-nfl-google.csv", header=1)
nfl_trends.head()
# convert percent strings into floats
numeric_data = (nfl_trends.iloc[:,1:]
                .replace("%", "",regex=True)
                .astype(float))
numeric_data["DMA"] = nfl_trends["DMA"]
nfl_trends = numeric_data

# get the search interest for each league, normalizes
interests  = nfl_trends.iloc[:,:-2].values
interests = interests / interests.sum(1, keepdims=True)

from scipy.stats import entropy

def JSdiv(p, q):
    """Jensen-Shannon divergence."""
    m = (p + q) / 2
    return (entropy(p, m, base=2.0) + entropy(q, m, base=2.0)) / 2

# initialize the distance matrix
n = interests.shape[0]
dist = np.zeros((n,n))
# compute JS distance for all pairs
for i in range(n):
    for j in range(n):
        dist[i,j] = JSdiv(interests[i,:], interests[j,:])

from sklearn import manifold

# intiialize
MDS =  manifold.MDS(dissimilarity="precomputed")
# transform to lower dimensional representation with JS distance
lower = MDS.fit_transform(dist)

# intiialize
MDS =  manifold.MDS()
# transform to lower dimensional representation with Euclidean distance
lower_naive = MDS.fit_transform(interests)

# put into a dataframe and plot
lower_df = pd.DataFrame({"x_JS":lower[:,0], "y_JS":lower[:,1],
                         "x_naive":lower_naive[:,0], "y_naive": lower_naive[:,1],
                         "DMA":nfl_trends["DMA"], 
                        "Trump 2016 Vote%":nfl_trends["Trump 2016 Vote%"]})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,10))
s = ax1.scatter(lower_df["x_JS"], lower_df["y_JS"], c=lower_df["Trump 2016 Vote%"],
           cmap=plt.get_cmap("viridis"))
s = ax2.scatter(lower_df["x_naive"], lower_df["y_naive"], c=lower_df["Trump 2016 Vote%"],
           cmap=plt.get_cmap("viridis"))

ax1.set_title("JS Distance MDS Representation")
ax2.set_title("Euclidean MDS Representation")

cbar = fig.colorbar(s)
cbar.ax.set_title("2016 Trump Vote%");

