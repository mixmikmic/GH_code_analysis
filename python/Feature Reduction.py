import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

ds = pd.read_csv('feature_all.csv', index_col=0)
# drop feature that will not have impact 
# ds.drop(['solidity','euler_number' , 'extent'], 1 , inplace=True) 
ds

y = ds['class']
classifier_f = open('Y_labels.pickle','wb')
pickle.dump(y, classifier_f)
classifier_f.close()    

y
# df1 = ds[['solidity','euler_number' , 'extent']]
# df1

# compute mean and std for all class 1 samples
# sample - (all feature samples std)

ds_class_1 = ds[(ds['class' ] > 0)]

ds_class_1.loc[0:,'Mean'] = (ds_class_1['Mean']) / ds_class_1['Mean'].std()
ds_class_1.loc[0:,'ASM'] = (ds_class_1['ASM']  / ds_class_1['ASM'].std()) 
ds_class_1.loc[0:,'contrast'] = (ds_class_1['contrast']) / ds_class_1['contrast'].std()
ds_class_1.loc[0:,'correlation'] = (ds_class_1['correlation']  /  ds_class_1['correlation'].std()) 
ds_class_1.loc[0:,'dissimilarity'] = (ds_class_1['dissimilarity'] / ds_class_1['dissimilarity'].std())
ds_class_1.loc[0:,'energy'] = (ds_class_1['energy']  /  ds_class_1['energy'].std())
ds_class_1.loc[0:,'kurtosis'] = (ds_class_1['kurtosis'] /  ds_class_1['kurtosis'].std())
ds_class_1.loc[0:,'skew'] = (ds_class_1['skew']  / ds_class_1['skew'].std())
ds_class_1.loc[0:,'Standard deviation'] = (ds_class_1['Standard deviation']  / ds_class_1['Standard deviation'].std())
ds_class_1.loc[0:,'area'] = (ds_class_1['area']) / ds_class_1['area'].std()
ds_class_1.loc[0:,'homogeneity'] = (ds_class_1['homogeneity']  / ds_class_1['homogeneity'].std()) 
ds_class_1.loc[0:,'orientation'] = (ds_class_1['orientation']  /  ds_class_1['orientation'].std()) 
ds_class_1.loc[0:,'convex_area'] = (ds_class_1['convex_area'] / ds_class_1['convex_area'].std())
ds_class_1.loc[0:,'eccentricity'] = (ds_class_1['eccentricity'] /  ds_class_1['eccentricity'].std())

ds_class_1.head()

# old Version
# ds_class_1['Mean'] = (ds_class_1['Mean'] - ds_class_1['Mean'].mean()) / ds_class_1['Mean'].std()

# compute mean and std for all class( 0 )amples
# sample - (all feature samples std

ds_class_0= ds[(ds['class' ] < 1 )]

ds_class_0.loc[0:,'Mean'] = (ds_class_0['Mean']) / ds_class_0['Mean'].std()
ds_class_0.loc[0:,'ASM'] = (ds_class_0['ASM']  / ds_class_0['ASM'].std()) 
ds_class_0.loc[0:,'contrast'] = (ds_class_0['contrast']) / ds_class_0['contrast'].std()
ds_class_0.loc[0:,'correlation'] = (ds_class_0['correlation']  /  ds_class_0['correlation'].std()) 
ds_class_0.loc[0:,'dissimilarity'] = (ds_class_0['dissimilarity'] / ds_class_0['dissimilarity'].std())
ds_class_0.loc[0:,'energy'] = (ds_class_0['energy']  /  ds_class_0['energy'].std())
ds_class_0.loc[0:,'kurtosis'] = (ds_class_0['kurtosis'] /  ds_class_0['kurtosis'].std())
ds_class_0.loc[0:,'skew'] = (ds_class_0['skew']  / ds_class_0['skew'].std())
ds_class_0.loc[0:,'Standard deviation'] = (ds_class_0['Standard deviation']  / ds_class_0['Standard deviation'].std())
ds_class_0.loc[0:,'area'] = (ds_class_0['area']) / ds_class_0['area'].std()
ds_class_0.loc[0:,'homogeneity'] = (ds_class_0['homogeneity']  / ds_class_0['homogeneity'].std()) 
ds_class_0.loc[0:,'orientation'] = (ds_class_0['orientation']  /  ds_class_0['orientation'].std()) 
ds_class_0.loc[0:,'convex_area'] = (ds_class_0['convex_area'] / ds_class_0['convex_area'].std())
ds_class_0.loc[0:,'eccentricity'] = (ds_class_0['eccentricity'] /  ds_class_0['eccentricity'].std())
ds_class_0

# concating all  features
frames = [ds_class_1 , ds_class_0]
ds_new = pd.concat(frames, ignore_index=True)
ds_new.to_csv('Feature_all_optomized.csv')

# feature selection with regression for texture feature only
from sklearn.linear_model import Lasso

df = pd.read_csv('brain-cancer.data')
# replace (?) with -99999 in data set 
df.replace('?', -99999, inplace=True )
# # remove  id colunm cause it has no effect in learning
df.drop(['id'],1, inplace=True)
feature = df.drop(['class'],1)
labels =df['class'] 
classifier_f = open('n_y.pickle','wb')
pickle.dump(labels, classifier_f)
classifier_f.close()    
regression = Lasso(alpha=0.1)
regression.fit(feature, labels)
# regression.predict([[0.62169290804739175,0.92176444484776776, 1]])
print(regression.coef_)
print(regression.intercept_)

# feature = df[['homogeneity', 'correlation','convex_area']]
# labels =df['class'] 
# # print(labels)
# print(feature)
regression = Lasso(alpha=0.1)
regression.fit(feature, labels)
regression.predict([[0.62169290804739175,0.92176444484776776, 1]])
print(regression.coef_)
print(regression.intercept_)

# PCA
from sklearn.decomposition import PCA , IncrementalPCA

def do_pca(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    return pca

data = np.array(feature)
pca = do_pca(data)
print(pca.explained_variance_ratio_)
first_pc = pca.components_[0]
second_pc = pca.components_[1]

transformed_data = pca.transform(data)
# fil = open("pca_trans_data.pickle", 'wb')
# pickle.dump( transformed_data, fil)
# fil.close()

for ii , jj in zip(transformed_data, data):
    plt.scatter( first_pc[0] * ii[0], first_pc[1]*ii[0], color='r' )
    plt.scatter( second_pc[0] * ii[1], second_pc[1]*ii[1], color='b' )
    plt.scatter(jj[0], jj[1], color='c')
    

plt.show()



# IncrementalPCA
ipca =  IncrementalPCA(n_components=3)
print(ipca)
ipca.fit(feature)
x =ipca.transform(feature) 
print(ipca.explained_variance_ratio_)

# ipca.get_covariance()
ipca.get_precision()
ipca.get_params(deep=True)

from sklearn import random_projection

transform  = random_projection.GaussianRandomProjection(n_components=5)
feature_new = transform.fit_transform(feature)
classifier_f = open('n_GaussianRandomProjection.pickle','wb')
pickle.dump(feature_new, classifier_f)
classifier_f.close()    
print(feature_new)



# FeatureAgglomeration

from sklearn.cluster import FeatureAgglomeration

agglo = FeatureAgglomeration( n_clusters=2)

agglo.fit(feature)
X_reduced = agglo.transform(feature)
# X_reduced
agglo.n_components_



