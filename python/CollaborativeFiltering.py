import numpy as np
import scipy as sc
from pandas import Series,DataFrame
import pandas as pd

from scipy import spatial
from sklearn import preprocessing

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from collections import OrderedDict
from fractions import Fraction

get_ipython().magic('matplotlib inline')
mpl.rcParams['figure.figsize'] = (10.0, 5)

df = pd.DataFrame({'Hotel1' :[1,0,3,0,0,5,0,0,5,0,4,0],
                   'Hotel2' :[0,0,5,4,0,0,4,0,0,2,1,3],
                   'Hotel3' :[2,4,0,1,2,0,3,0,4,3,5,0],
                   'Hotel4' :[0,2,4,0,5,0,0,4,0,0,2,0],
                   'Hotel5' :[0,0,4,3,4,2,0,0,0,0,2,5],
                   'Hotel6' : [1,0,3,0,3,0,0,2,0,0,4,0],
                   }, index=['User1','User2','User3','User4','User5',
                             'User6','User7','User8','User9','User10','User11','User12'])
df = df.transpose()
df

# check if hotels have enough ratings (enough support) to be able to make predictions
df.transpose().plot.barh(stacked=True)

# find 0 values
no_rating_mask = (df == 0)
no_rating_mask

#comes after
#df[no_rating_mask] = None
#df

# possibility 2 to find hotel rating mean values
hotel_rating_averages = df[np.invert(no_rating_mask)].mean(axis=1)
hotel_rating_averages

# normalise dataset
dfn = df.sub(hotel_rating_averages, axis=0)
dfn = dfn.round(1)
dfn

# put 0 values where no values was found
dfn[no_rating_mask] = 0
# and round values
dfn = dfn.round(1)

dfn

# inspect hotel similarities 
sns.pairplot(dfn.transpose())

# we could also plot hotel recommendation values vectors 
soa = dfn.transpose().values
print zip(*soa)
X,Y,U,V,Z,E = zip(*soa)
plt.figure()
ax = plt.gca()
ax.quiver(X,Y,U,V,Z,E, angles='xy',scale_units='xy',scale=1)
ax.set_xlim([-1,4])
ax.set_ylim([-1,1])
plt.draw()
plt.show()

# pearson correlation similarity
# option 1
hotel_similarity_df = dfn.transpose().corr().round(2)
hotel_similarity_df

sns.heatmap(hotel_similarity_df, annot=True)

# we couuld also calculate hotel similarities this way
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

A_sparse = sparse.csr_matrix(dfn.as_matrix())

#also can output sparse matrices
similarities_sparse = cosine_similarity(A_sparse, dense_output=False)
print('hotel pairwise similarity:\n {}\n'.format(similarities_sparse))

# Hotel1 is most similar to the Hotels 3 and 6
mask = hotel_similarity_df["Hotel1"] > 0.30
mask

# take ratings of most similar hotels (3 and 6)
hotel_ratings = df.User5[mask].values[1:]
hotel_ratings

# take similarities of most similar hotels (3 and 6)
hotel_sim = hotel_similarity_df.Hotel1[mask].values[1:]
hotel_sim

#calculate rating for hotel 1 from user 5
# predict by taking weighted average
r_15 = sum(hotel_ratings * hotel_sim) / sum(hotel_sim)
print "User 5 would rate Hotel 1 with: ", round(r_15,1), " stars"

