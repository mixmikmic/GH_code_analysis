# http://scikit-learn.org/stable/modules/decomposition.html
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html
# https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
# https://intoli.com/blog/pca-and-svd/
# https://github.com/jakevdp/sklearn_tutorial/blob/master/notebooks/04.1-Dimensionality-PCA.ipynb
# http://sebastianraschka.com/Articles/2014_python_lda.html#a-comparison-of-pca-and-lda
# http://sebastianraschka.com/Articles/2014_pca_step_by_step.html


# PCA vs SVD https://math.stackexchange.com/questions/3869/what-is-the-intuitive-relationship-between-svd-and-pca

# import the Iris dataset from scikit-learn
from sklearn.datasets import load_iris
# import our plotting module
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# load the Iris dataset
iris = load_iris()

# seperate the features and response variable
iris_X, iris_y = iris.data, iris.target

# the names of the flower we are trying to predict.
iris.target_names

# Names of the features
iris.feature_names

# for labelling: {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
label_dict = {i: k for i, k in enumerate(iris.target_names)}

def plot(X, y, title, x_label, y_label):
    ax = plt.subplot(111)
    for label,marker,color in zip(
    range(3),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X[:,0].real[y == label],
            y=X[:,1].real[y == label],
            color=color,
            alpha=0.5,
            label=label_dict[label]
            )

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)

plot(iris_X, iris_y, "Original Iris Data", "sepal length (cm)", "sepal width (cm)")

# Calculate a PCA manually

# import numpy
import numpy as np

# calculate the mean vector
mean_vector = iris_X.mean(axis=0)
print mean_vector

# calculate the covariance matrix
cov_mat = np.cov((iris_X).T)
print cov_mat.shape

cov_mat

# calculate the eigenvectors and eigenvalues of our covariance matrix of the iris dataset
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

# Print the eigen vectors and corresponding eigenvalues
# in order of descending eigenvalues
for i in range(len(eig_val_cov)):
    eigvec_cov = eig_vec_cov[:,i]
    print 'Eigenvector {}: \n{}'.format(i+1, eigvec_cov)
    print 'Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i])
    print 30 * '-'

# the percentages of the variance captured by each eigenvalue
# is equal to the eigenvalue of that components divided by
# the sum of all eigen values
explained_variance_ratio = eig_val_cov/eig_val_cov.sum()
explained_variance_ratio

# Scree Plot

plt.plot(np.cumsum(explained_variance_ratio))
plt.title('Scree Plot')
plt.xlabel('Principal Component (k)')
plt.ylabel('% of Variance Explained <= k')

# store the top two eigenvectors in a variable
top_2_eigenvectors = eig_vec_cov[:,:2].T

# show the transpose so that each row is a principal component, we have two rows == two components
top_2_eigenvectors

# to transform our data from having shape (150, 4) to (150, 2)
# we will multiply the matrices of our data and our eigen vectors together
# notice how I am centering the data first. I am doing this to replicate how scikit-learn PCA's algorithm functions
np.dot(iris_X, top_2_eigenvectors.T)[:5,]

# scikit-learn's version of PCA
from sklearn.decomposition import PCA

# Like any other sklearn module, we first instantiate the class
pca = PCA(n_components=2)

# fit the PCA to our data
pca.fit(iris_X)

pca.components_
# note that the second column is the negative of the manual process
# this is because eignevectors can be positive or negative
# It should have little to no effect on our machine learning pipelines

# sklearn PCA centers the data first while transforming, so these numbers won't match our manual process.
pca.transform(iris_X)[:5,]

# manually centering our data to match scikit-learn's implementation of PCA
np.dot(iris_X-mean_vector, top_2_eigenvectors.T)[:5,]



# Plot the original and projected data
plot(iris_X, iris_y, "Original Iris Data", "sepal length (cm)", "sepal width (cm)")
plt.show()
plot(pca.transform(iris_X), iris_y, "Iris: Data projected onto first two PCA components", "PCA1", "PCA2")

# percentage of variance in data explained by each component
# same as what we calculated earlier

pca.explained_variance_ratio_





# show how pca attempts to eliminate dependence between columns

# capture all four principal components
full_pca = PCA(n_components=4)

# fit our PCA to the iris dataset
full_pca.fit(iris_X)

# show the correlation matrix of the original dataset
np.corrcoef(iris_X.T)

# correlation coefficients above the diagonal
np.corrcoef(iris_X.T)[[0, 0, 0, 1, 1], [1, 2, 3, 2, 3]]

# average correlation of original iris dataset.
np.corrcoef(iris_X.T)[[0, 0, 0, 1, 1], [1, 2, 3, 2, 3]].mean()

pca_iris = full_pca.transform(iris_X)
# average correlation of PCAed iris dataset.
np.corrcoef(pca_iris.T)[[0, 0, 0, 1, 1], [1, 2, 3, 2, 3]].mean()
# VERY close to 0 because columns are independent from one another
# This is an important consequence of performing an eigen value decomposition



# import our scaling module
from sklearn.preprocessing import StandardScaler
# center our data, not a full scaling
X_centered = StandardScaler(with_std=False).fit_transform(iris_X)

X_centered[:5,]

# Plot our centered data
plot(X_centered, iris_y, "Iris: Data Centered", "sepal length (cm)", "sepal width (cm)")



# fit our PCA (with n_components still set to 2) on our centered data
pca.fit(X_centered)

# same components as before
pca.components_  

# same projection when data are centered because PCA does this automatically
pca.transform(X_centered)[:5,]  

# Plot PCA projection of centered data, same as previous PCA projected data
plot(pca.transform(X_centered), iris_y, "Iris: Data projected onto first two PCA components with centered data", "PCA1", "PCA2")

# percentage of variance in data explained by each component

pca.explained_variance_ratio_



# doing a normal z score scaling
X_scaled = StandardScaler().fit_transform(iris_X)

# Plot scaled data
plot(X_scaled, iris_y, "Iris: Data Scaled", "sepal length (cm)", "sepal width (cm)")



# fit our 2-dimensional PCA on our scaled data
pca.fit(X_scaled)

# different components as cenetered data
pca.components_

# different projection when data are scaled
pca.transform(X_scaled)[:5,]  

# percentage of variance in data explained by each component
pca.explained_variance_ratio_

# Plot PCA projection of scaled data
plot(pca.transform(X_scaled), iris_y, "Iris: Data projected onto first two PCA components", "PCA1", "PCA2")





# how to interpret and use components
pca.components_  # a 2 x 4 matrix

first_scaled_flower

# Multiply original matrix (150 x 4) by components transposed (4 x 2) to get new columns (150 x 2)
np.dot(X_scaled, pca.components_.T)[:5,]

# extract the first row of our scaled data
first_scaled_flower = X_scaled[0]
# extract the two PC's
first_Pc = pca.components_[0]
second_Pc = pca.components_[1]

first_scaled_flower.shape  # (4,)

# same result as the first row of our matrix multiplication
np.dot(first_scaled_flower, first_Pc), np.dot(first_scaled_flower, second_Pc)

# This is how the transform method works in pca
pca.transform(X_scaled)[:5,]





# visualize PCA components

# cut out last two columns of the original iris dataset
iris_2_dim = iris_X[:,2:4]

# center the data
iris_2_dim = iris_2_dim - iris_2_dim.mean(axis=0)

plot(iris_2_dim, iris_y, "Iris: Only 2 dimensions", "sepal length", "sepal width")

# instantiate a PCA of 2 components
twodim_pca = PCA(n_components=2)

# fit and transform our truncated iris data
iris_2_dim_transformed = twodim_pca.fit_transform(iris_2_dim)



plot(iris_2_dim_transformed, iris_y, "Iris: PCA performed on only 2 dimensions", "PCA1", "PCA2")

# This code is graphing both the original iris data and the projected version of it using PCA.
# Moreover, on each graph, the principal components are graphed as vectors on the data themselves
# The longer of the arrows is meant to describe the first principal component and
# the shorter of the arrows describes the second principal component
def draw_vector(v0, v1, ax):
    arrowprops=dict(arrowstyle='->',linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
    
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

# plot data
ax[0].scatter(iris_2_dim[:, 0], iris_2_dim[:, 1], alpha=0.2)
for length, vector in zip(twodim_pca.explained_variance_, twodim_pca.components_):
    v = vector * np.sqrt(length)  # elongdate vector to match up to explained_variance
    draw_vector(twodim_pca.mean_, 
                twodim_pca.mean_ + v, ax=ax[0])
ax[0].set(xlabel='x', ylabel='y', title='Original Iris Dataset',
         xlim=(-3, 3), ylim=(-2, 2))


ax[1].scatter(iris_2_dim_transformed[:, 0], iris_2_dim_transformed[:, 1], alpha=0.2)
for length, vector in zip(twodim_pca.explained_variance_, twodim_pca.components_):
    transformed_component = twodim_pca.transform([vector])[0]  # transform components to new coordinate system
    v = transformed_component * np.sqrt(length)  # elongdate vector to match up to explained_variance
    draw_vector(iris_2_dim_transformed.mean(axis=0),
                iris_2_dim_transformed.mean(axis=0) + v, ax=ax[1])
ax[1].set(xlabel='component 1', ylabel='component 2',
          title='Projected Data',
          xlim=(-3, 3), ylim=(-1, 1))



# LDA is better than PCA for classification

# calculate the mean for each class
# to do this we will separate the iris dataset into three dataframes
# one for each flower, then we will take one's mean columnwise
mean_vectors = []
for cl in [0, 1, 2]:
    class_mean_vector = np.mean(iris_X[iris_y==cl], axis=0)
    mean_vectors.append(class_mean_vector)
    print label_dict[cl], class_mean_vector

# Calculate within-class scatter matrix
S_W = np.zeros((4,4))
# for each flower
for cl,mv in zip([0, 1, 2], mean_vectors):
    # scatter matrix for every class, starts with all 0's
    class_sc_mat = np.zeros((4,4))  
    # for each row that describes the specific flower
    for row in iris_X[iris_y == cl]:
        # make column vectors 
        row, mv = row.reshape(4,1), mv.reshape(4,1) 
        # this is a 4x4 matrix
        class_sc_mat += (row-mv).dot((row-mv).T)
    # sum class scatter matrices
    S_W += class_sc_mat          
    
S_W

# calculate the between-class scatter matrix

# mean of entire dataset
overall_mean = np.mean(iris_X, axis=0).reshape(4,1)

# will eventually become between class scatter matrix
S_B = np.zeros((4,4))
for i,mean_vec in enumerate(mean_vectors):  
    # number of flowers in each species
    n = iris_X[iris_y==i,:].shape[0]
    # make column vector for each specied
    mean_vec = mean_vec.reshape(4,1)
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

S_B

# calculate eigenvalues and eigenvectors of S−1W x SB
eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(S_W), S_B))
eig_vecs = eig_vecs.real
eig_vals = eig_vals.real

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i]
    print 'Eigenvector {}: {}'.format(i+1, eigvec_sc)
    print 'Eigenvalue {:}: {}'.format(i+1, eig_vals[i])
    print

# keep the top two linear discriminants
linear_discriminants = eig_vecs.T[:2]

linear_discriminants

#explained variance ratios
eig_vals / eig_vals.sum()

# LDA projected data

lda_iris_projection = np.dot(iris_X, linear_discriminants.T)

lda_iris_projection[:5,]

plot(lda_iris_projection, iris_y, "LDA Projection", "LDA1", "LDA2")





from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# instantiate the LDA module
lda = LinearDiscriminantAnalysis(n_components=2)

# fit and transform our original iris data
X_lda_iris = lda.fit_transform(iris_X, iris_y)

# plot the projected data
plot(X_lda_iris, iris_y, "LDA Projection", "LDA1", "LDA2")

# show that the sklearn components are just a scalar multiplication from the manual components we calculateda
for manual_component, sklearn_component in zip(eig_vecs.T[:2], lda.scalings_.T):
    print sklearn_component / manual_component

# same as manual calculations
lda.explained_variance_ratio_

# essentially the same as pca.components_, but transposed (4x2 instead of 2x4)
lda.scalings_  



# fit our LDA to scaled data
X_lda_iris = lda.fit_transform(X_scaled, iris_y)



lda.scalings_  # different scalings when data are scaled

# LDA1 is the best axis for SEPERATING the classes

# fit our LDA to our truncated iris dataset
iris_2_dim_transformed_lda = lda.fit_transform(iris_2_dim, iris_y)

# project data
iris_2_dim_transformed_lda[:5,]

# different notation
components = lda.scalings_.T  # transposing to get same usage as PCA. I want the rows to be our components
print components

np.dot(iris_2_dim, components.T)[:5,]  # same as transform method

np.corrcoef(iris_2_dim.T)  # original features are highly correllated

# new LDA features are highly uncorrellated, like in PCA
np.corrcoef(iris_2_dim_transformed_lda.T)  

# This code is graphing both the original iris data and the projected version of it using LDA.
# Moreover, on each graph, the scalings of the LDA are graphed as vectors on the data themselves
# The longer of the arrows is meant to describe the first scaling vector and
# the shorter of the arrows describes the second scaling vector
def draw_vector(v0, v1, ax):
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
    
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

# plot data
ax[0].scatter(iris_2_dim[:, 0], iris_2_dim[:, 1], alpha=0.2)
for length, vector in zip(lda.explained_variance_ratio_, components):
    v = vector * .5
    draw_vector(lda.xbar_, lda.xbar_ + v, ax=ax[0])  # lda.xbar_ is equivalent to pca.mean_
ax[0].axis('equal')
ax[0].set(xlabel='x', ylabel='y', title='Original Iris Dataset',
         xlim=(-3, 3), ylim=(-3, 3))

ax[1].scatter(iris_2_dim_transformed_lda[:, 0], iris_2_dim_transformed_lda[:, 1], alpha=0.2)
for length, vector in zip(lda.explained_variance_ratio_, components):
    transformed_component = lda.transform([vector])[0]
    v = transformed_component * .1
    draw_vector(iris_2_dim_transformed_lda.mean(axis=0), iris_2_dim_transformed_lda.mean(axis=0) + v, ax=ax[1])
ax[1].axis('equal')
ax[1].set(xlabel='lda component 1', ylabel='lda component 2',
          title='Linear Discriminant Analysis Projected Data',
          xlim=(-10, 10), ylim=(-3, 3))

# notice how the component, instead of going with the variance of the data
# goes almost perpendicular to it, its following the seperation of the classes instead
# note how its almost parallel with the gap between the flowers on the left and right side
# LDA is trying to capture the separation between classes



from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Create a PCA module to keep a single component
single_pca = PCA(n_components=1)

# Create a LDA module to keep a single component
single_lda = LinearDiscriminantAnalysis(n_components=1)

# Instantiate a KNN model
knn = KNeighborsClassifier(n_neighbors=3)



# run a cross validation on the KNN without any feature transformation
knn_average = cross_val_score(knn, iris_X, iris_y).mean()

# This is a baseline accuracy. If we did nothing, KNN on its own achieves a 98% accuracy
knn_average

get_ipython().run_cell_magic('timeit', '', 'knn_average = cross_val_score(knn, iris_X, iris_y).mean()')

# create a pipeline that performs PCA
pca_pipeline = Pipeline([('pca', single_pca), ('knn', knn)])

pca_average = cross_val_score(pca_pipeline, iris_X, iris_y).mean()

pca_average

get_ipython().run_cell_magic('timeit', '', 'cross_val_score(pca_pipeline, iris_X, iris_y).mean()')

lda_pipeline = Pipeline([('lda', single_lda), ('knn', knn)])

lda_average = cross_val_score(lda_pipeline, iris_X, iris_y).mean()

# better prediction accuracy than PCA by a good amount, but not as good as original
lda_average

get_ipython().run_cell_magic('timeit', '', 'cross_val_score(lda_pipeline, iris_X, iris_y).mean()')



# LDA is much better at creating axes for classification purposes

# try LDA with 2 components
lda_pipeline = Pipeline([('lda', LinearDiscriminantAnalysis(n_components=2)), 
                         ('knn', knn)])

lda_average = cross_val_score(lda_pipeline, iris_X, iris_y).mean()

# Just as good as using original data
lda_average

get_ipython().run_cell_magic('timeit', '', 'cross_val_score(lda_pipeline, iris_X, iris_y).mean()')



# compare our feature transformation tools to a feature selection tool
from sklearn.feature_selection import SelectKBest
# try all possible values for k, excluding keeping all columns
for k in [1, 2, 3]:
    # make the pipeline
    select_pipeline = Pipeline([('select', SelectKBest(k=k)), ('knn', knn)])
    # cross validate the pipeline
    select_average = cross_val_score(select_pipeline, iris_X, iris_y).mean()
    print k, "best feature has accuracy:", select_average
    
# LDA is even better than the best selectkbest

get_ipython().run_cell_magic('timeit', '', 'cross_val_score(select_pipeline, iris_X, iris_y).mean()')





def get_best_model_and_accuracy(model, params, X, y):
    grid = GridSearchCV(model,           # the model to grid search
                        params,          # the parameter set to try 
                        error_score=0.)  # if a parameter set raises an error, continue and set the performance as a big, fat 0
    grid.fit(X, y)           # fit the model and parameters
    # our classical metric for performance
    print "Best Accuracy: {}".format(grid.best_score_)
    # the best parameters that caused the best accuracy
    print "Best Parameters: {}".format(grid.best_params_)
    # the average time it took a model to fit to the data (in seconds)
    print "Average Time to Fit (s): {}".format(round(grid.cv_results_['mean_fit_time'].mean(), 3))
    # the average time it took a model to predict out of sample data (in seconds)
    # this metric gives us insight into how this model will perform in real-time analysis
    print "Average Time to Score (s): {}".format(round(grid.cv_results_['mean_score_time'].mean(), 3))

from sklearn.model_selection import GridSearchCV
iris_params = {
                'preprocessing__scale__with_std': [True, False],
                'preprocessing__scale__with_mean': [True, False],
                'preprocessing__pca__n_components':[1, 2, 3, 4], 
                
                # according to scikit-learn docs, max allowed n_components for LDA is number of classes - 1
                'preprocessing__lda__n_components':[1, 2],  
                
                'clf__n_neighbors': range(1, 9)
              }
# make a larger pipeline
preprocessing = Pipeline([('scale', StandardScaler()), 
                          ('pca', PCA()), 
                          ('lda', LinearDiscriminantAnalysis())])


iris_pipeline = Pipeline(steps=[('preprocessing', preprocessing), 
                                ('clf', KNeighborsClassifier())])

get_best_model_and_accuracy(iris_pipeline, iris_params, iris_X, iris_y)

# can't use PCA on sparse data..

# http://scikit-learn.org/stable/modules/decomposition.html
# https://www.kaggle.com/datasf/case-data-from-san-f

import pandas as pd

hotel_reviews = pd.read_csv('../data/7282_1.csv')

hotel_reviews.shape

hotel_reviews.head()

# Let's only include reviews from the US to try to only include english reviews

# plot the lats and longs of reviews
hotel_reviews.plot.scatter(x='longitude', y='latitude')

#Filter to only include datapoints within the US
hotel_reviews = hotel_reviews[((hotel_reviews['latitude']<=50.0) & (hotel_reviews['latitude']>=24.0)) & ((hotel_reviews['longitude']<=-65.0) & (hotel_reviews['longitude']>=-122.0))]

# Plot the lats and longs again
hotel_reviews.plot.scatter(x='longitude', y='latitude')
# Only looking at reviews that are coming from the US

hotel_reviews.shape

texts = hotel_reviews['reviews.text']

# import the sentence tokenizer from nltk
from nltk.tokenize import sent_tokenize
sent_tokenize("hello! I am Sinan. How are you??? I am fine")

sentences = reduce(lambda x, y:x+y, texts.apply(lambda x: sent_tokenize(str(x).decode('utf-8'))))

# the number of sentences
len(sentences)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')

tfidf_transformed = tfidf.fit_transform(sentences)

tfidf_transformed

# try to fit PCA

PCA(n_components=1000).fit(tfidf_transformed)

# can't work because it has to calculate a covariance matrix and to do that, the matrix needs to be dense

# we use another method in sklearn called Truncated SVD
# Truncated SVD uses a matrix trick to obtain the same components as PCA (when the data are scaled)
# and can work with sparse matrices

# components are a not exactly equal but they are up to a very precise decimal

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=2)
pca = PCA(n_components=2)

# check if components of PCA and TruncatedSVD are same for a dataset
# by substracting the two matricies and seeing if, on average, the elements are very close to 0
print (pca.fit(iris_X).components_ - svd.fit(iris_X).components_).mean()  # not close to 0
# matrices are NOT the same

# check if components of PCA and TruncatedSVD are same for a centered dataset
print (pca.fit(X_centered).components_ - svd.fit(X_centered).components_).mean()  # close to 0
# matrices ARE the same

# check if components of PCA and TruncatedSVD are same for a scaled dataset
print (pca.fit(X_scaled).components_ - svd.fit(X_scaled).components_).mean()  # close to 0
# matrices ARE the same

(pca.fit(X_centered).components_ - svd.fit(X_centered).components_).mean()

svd = TruncatedSVD(n_components=1000)
svd.fit(tfidf_transformed)

# Scree Plot

plt.plot(np.cumsum(svd.explained_variance_ratio_))

# 1,000 components captures about 30% of the variance



# latent semantic analysis is a name given to the process of doing an SVD on sparse text document-term matricies
# It is done to find latent structure in text for the purposes of classification, clustering, etc

from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
svd = TruncatedSVD(n_components=10)
normalizer = Normalizer()

lsa = Pipeline(steps=[('tfidf', tfidf), ('svd', svd), ('normalizer', normalizer)])

lsa.fit(sentences)

lsa_sentences = lsa.transform(sentences)

lsa_sentences.shape

cluster = KMeans(n_clusters=10)

cluster.fit(lsa_sentences)



get_ipython().run_cell_magic('timeit', '', '# time it takes to cluster on the original document-term matrix of shape (118151, 280901)\ncluster.fit(tfidf_transformed)')

get_ipython().run_cell_magic('timeit', '', '# also time the prediction phase of the Kmeans clustering\ncluster.predict(tfidf_transformed)')



get_ipython().run_cell_magic('timeit', '', '# time the time to cluster after latent semantic analysis of shape (118151, 10)\ncluster.fit(lsa_sentences)\n# over 80 times faster than fitting on the original tfidf dataset')

get_ipython().run_cell_magic('timeit', '', '# also time the prediction phase of the Kmeans clustering after LSA was performed\ncluster.predict(lsa_sentences)\n# over 4 times faster than predicting on the original tfidf dataset')



# transform texts to a cluster distance space
# each row represents an obsercation
cluster.transform(lsa_sentences).shape

predicted_cluster = cluster.predict(lsa_sentences)
predicted_cluster

# Distribution of "topics"
pd.Series(predicted_cluster).value_counts(normalize=True)

# create DataFrame of texts and predicted topics
texts_df = pd.DataFrame({'text':sentences, 'topic':predicted_cluster})

texts_df.head()

print "Top terms per cluster:"
original_space_centroids = svd.inverse_transform(cluster.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = lsa.steps[0][1].get_feature_names()
for i in range(10):
    print "Cluster %d:" % i
    print ', '.join([terms[ind] for ind in order_centroids[i, :5]])
    print 

lsa.steps[0][1]





from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from time import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

get_ipython().run_line_magic('matplotlib', 'inline')

# load the dataset
# the optional parameter: min_faces_per_person: 
# will only retain pictures of people that have at least min_faces_per_person different pictures.
# the optional parameter: resize is the ratio used to resize the each face picture.
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
y = lfw_people.target
n_features = X.shape[1]

X.shape

# plot one of the faces
plt.imshow(X[0].reshape((h, w)), cmap=plt.cm.gray)
lfw_people.target_names[y[0]]

# plot one of the faces
plt.imshow(StandardScaler().fit_transform(X)[0].reshape((h, w)), cmap=plt.cm.gray)
lfw_people.target_names[y[0]]

# let's plot another face
plt.imshow(X[100].reshape((h, w)), cmap=plt.cm.gray)
lfw_people.target_names[y[100]]

# the label to predict is the id of the person
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# let's split our dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Compute a PCA (eigenfaces) on the face dataset 
n_components = 200

"""
from sklearn docs:

The optional parameter whiten=True makes it possible to project the data onto the singular space 
while scaling each component to unit variance. This is often useful if the models down-stream make strong 
assumptions on the isotropy of the signal: this is for example the case for 
Support Vector Machines with the RBF kernel and the K-Means clustering algorithm.
"""

# instantiate the PCA module
pca = PCA(n_components=n_components, whiten=True)

# create a pipeline called preprocessing that will scale data and then apply PCA
preprocessing = Pipeline([('scale', StandardScaler()), ('pca', pca)])

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))

# fit the pipeline to the training set
preprocessing.fit(X_train)

# grab the PCA from the pipeline
extracted_pca = preprocessing.steps[1][1]

# take the components from the PCA ( just like we did with iris )
# and reshape them to have the same height and weight as the original photos
eigenfaces = extracted_pca.components_.reshape((n_components, h, w))

# Scree Plot

plt.plot(np.cumsum(extracted_pca.explained_variance_ratio_))

# starting at 100 components captures over 90% of the variance compared to the 1,850 original features

# This function is meant to plot several images in a gallery with given titles
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()



# Use a pipeline to make this process easier
logreg = LogisticRegression()

# create the pipeline
face_pipeline = Pipeline(steps=[('preprocessing', preprocessing), ('logistic', logreg)])

print "fitting preprocessing pipeline to X_train and transforming X"

pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints a more readable confusion matrix with heat labels and options for noramlization
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



param_grid = {'C': [1e-2, 1e-1,1e0,1e1, 1e2]}



# fit without using PCA to see what the difference will be
t0 = time()

clf = GridSearchCV(logreg, param_grid)
clf = clf.fit(X_train, y_train)
best_clf = clf.best_estimator_

# Predicting people's names on the test set
y_pred = best_clf.predict(X_test)

print accuracy_score(y_pred, y_test), "Accuracy score for best estimator"
print(classification_report(y_test, y_pred, target_names=target_names))
print plot_confusion_matrix(confusion_matrix(y_test, y_pred, labels=range(n_classes)), target_names)
print round((time() - t0), 1), "seconds to grid search and predict the test set"

# now fit with PCA to see if our accuracy improves
t0 = time()

clf = GridSearchCV(logreg, param_grid)
clf = clf.fit(X_train_pca, y_train)
best_clf = clf.best_estimator_

# Predicting people's names on the test set
y_pred = best_clf.predict(X_test_pca)

print accuracy_score(y_pred, y_test), "Accuracy score for best estimator"
print(classification_report(y_test, y_pred, target_names=target_names))
print plot_confusion_matrix(confusion_matrix(y_test, y_pred, labels=range(n_classes)), target_names)
print round((time() - t0), 1), "seconds to grid search and predict the test set"





# get a list of predicted names and true names to plot with faces in test set
prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

# splot a sample of the test set with predicted and true names
plot_gallery(X_test, prediction_titles, h, w)



# Create a larger pipeline to gridsearch
face_params = {'logistic__C':[1e-2, 1e-1, 1e0, 1e1, 1e2], 
               'preprocessing__pca__n_components':[100, 150, 200, 250, 300],
               'preprocessing__pca__whiten':[True, False],
               'preprocessing__lda__n_components':range(1, 7)  
               # [1, 2, 3, 4, 5, 6] recall the max allowed is n_classes-1
              }

pca = PCA()
lda = LinearDiscriminantAnalysis()

preprocessing = Pipeline([('scale', StandardScaler()), ('pca', pca), ('lda', lda)])

logreg = LogisticRegression()
face_pipeline = Pipeline(steps=[('preprocessing', preprocessing), ('logistic', logreg)])

get_best_model_and_accuracy(face_pipeline, face_params, X, y)

# much better than original data and very fast to predict and train!



# talk about how these transformations are dope BUT they are predefined so we could learn new features 
# based on training data

# these predefined transformations might not work for a particular dataset 
# PCA is PCA no matter what dataset you choose to work with



