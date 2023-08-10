from sklearn.neighbors import NearestNeighbors 
from sklearn import datasets, neighbors
import numpy as np
# we will implement K-nearest neighbor search
# le's load  fisher iris dataset
iris_ = datasets.load_iris()

iris = iris_.data
def Knbor_Mat(X, K, t = 2.0, dist_metric = "euclidean", algorithm = "ball_tree"):
    
    n,p = X.shape
    
    knn = neighbors.NearestNeighbors(K+1, metric = dist_metric, algorithm=algorithm).fit(X)
    distances, nbors = knn.kneighbors(X)
    
    return(nbors[:,1:])

# calculation of reconstruction weights
from scipy import linalg

def get_weights(X, nbors, reg, K):
    
    n,p = X.shape
    
    Weights = np.zeros((n,n))
    
    for i in range(n):
        
        X_bors = X[nbors[i],:] - X[i]
        cov_nbors = np.dot(X_bors, X_bors.T)
        
        #regularization tems
        trace = np.trace(cov_nbors)
        if trace >0 :
            R = reg*trace
        else:
            R = reg
        
        cov_nbors.flat[::K+1] += R
        weights = linalg.solve(cov_nbors, np.ones(K).T, sym_pos=True)

        weights = weights/weights.sum()
        Weights[i, nbors[i]] = weights
        
    return(Weights)

# calculation of the new embedding
from scipy.linalg import eigh

def Y_(Weights,d):
    n,p = Weights.shape
    I = np.eye(n)
    m = (I-Weights)
    M = m.T.dot(m)
    
    eigvals, eigvecs = eigh(M, eigvals=(1, d), overwrite_a=True)
    ind = np.argsort(np.abs(eigvals))
    
    return(eigvecs[:, ind])
    

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

n_points = 1000
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)

def LLE_(X, K):
    reg =0.001
    nbors = Knbor_Mat(X,K)
    print(nbors[0])
    Weights = get_weights(X, nbors, reg,K)
    
    Y = Y_(Weights,2)
    return(Y)
test = [354,520, 246,134,3, 983, 186, 436, 893, 921]
def plotter(K):
    fig = plt.figure(figsize=(10,8))
    Y = LLE_(X , K)
    s = Y[test]
    plt.scatter(Y[:,0],Y[:,1],c=color, cmap=plt.cm.spectral)
    plt.scatter(s[:,0],s[:,1], c="black")

interact(plotter, K= widgets.IntSlider(min=10, max=100, value=10, step=10))

#necessary imports

from sklearn import datasets
from pyspark.sql import SQLContext as SQC
from pyspark.mllib.linalg import Vectors as mllibVs, VectorUDT as mllibVUDT
from pyspark.ml.linalg import Vectors as mlVs, VectorUDT as mlVUDT
from pyspark.sql.types import *
from pyspark.mllib.linalg import *
from pyspark.sql.functions import *
from pyspark.ml.feature import StandardScaler
from pyspark.mllib.linalg.distributed import IndexedRowMatrix
import math as m
import numpy as np

sqc = SQC(sc)
iris_tuple = datasets.load_iris()

iris = iris_tuple.data

df = (sqc.createDataFrame(sc.parallelize(iris.tolist()).zipWithIndex().
                              map(lambda x: (x[1], mlVs.dense(x[0]))), ["id", "features"]))    



udf_dist = udf(lambda x, y:  float(x.squared_distance(y)), DoubleType())


df_2 = df
    
df = df.crossJoin(df ).toDF('x_id', 'x_feature', 'y_id', 'y_feature')
df = df.withColumn("sim", udf_dist(df.x_feature, df.y_feature))

df = df.drop("x_feature")

st = struct([name for name in ["y_id", "sim","y_feature"]]).alias("map")
df = df.select("x_id", st)
df  = df.groupby("x_id").agg(collect_list("map").alias("map"))
df = df.join(df_2, df_2.id == df.x_id, "inner").drop("x_id")

# calculate local neighborhood matrix

def get_weights(map_list, k, features, reg):
    

    sorted_map = sorted(map_list, key = lambda x: x[1])[1:(k+1)]
    
    neighbors = np.array([s[2] for s in sorted_map])
    ind = [s[0] for s in sorted_map]
    
    nbors, n_features = neighbors.shape
    
    neighbors_mat = neighbors - features.toArray().reshape(1,n_features)
    
    cov_neighbors = np.dot(neighbors_mat, neighbors_mat.T)
    
    # add regularization term
    trace = np.trace(cov_neighbors)
    if trace > 0:
        R = reg * trace
    else:
        R = reg
    
    cov_neighbors.flat[::nbors + 1] += R
    
    weights = linalg.solve(cov_neighbors, np.ones(k).T, sym_pos=True)
    weights = weights/weights.sum()
    
    full_weights = np.zeros(len(map_list))
    full_weights[ind] = weights
    
    return(Vectors.dense(full_weights))

udf_sort = udf(get_weights, mllibVUDT())

df = df.withColumn("weights", udf_sort("map", lit(10), "features", lit(0.001)))

# udf for creating a row of identity matrix 
def I(ind):
    i = [0]*150
    i[ind]=1.0
    return(mllibVs.dense(i))

# convert dataframe to indexedrowmatrix
weights_irm = IndexedRowMatrix(df.select(["id","weights"]).rdd.map(lambda x:(x[0],  I(x[0])-x[1])))
M = weights_irm.toBlockMatrix().transpose().multiply( weights_irm.toBlockMatrix() )

SVD = M.toIndexedRowMatrix().computeSVD(150, True)

# select the vectors for low dimensional embedding
lle_embbeding = np.fliplr(SVD.V.toArray()[:,-(d+1):-1])

