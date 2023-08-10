from EM import* 
from time import time
from generate import GM_sample
get_ipython().magic('matplotlib inline')

def get_N_K_d(X,y):
    """ Get N, K and d given the data X and the true clusters vector y """  
    return X.shape[0], len(np.unique(y)), X.shape[1]

labels = [100,200,500,1000,2000,5000,10000]
n_labels = len(labels)

max_iter = 25
tol = -1
sample_size = 10

time_spark = np.zeros((n_labels,sample_size))
time_nospark = np.zeros((n_labels,sample_size))

for i,label in enumerate(labels):
    print(i)
    for j in range(sample_size):
        X = np.load("Datasets/dataset_"+str(label)+"_X.npy")
        y = np.load("Datasets/dataset_"+str(label)+"_y.npy")
        N,K,d = get_N_K_d(X,y)
        rdd = sc.parallelize(X)
        
        #NO SPARK
        EM_ = EM_noSpark()
        t = time()
        EM_.fit(X, n_clusters = K, max_iter = max_iter, criterion = None, verbose = False, tol=-1)
        time_nospark[i,j] = time()-t
        
        #SPARK
        EM = EM_Spark()
        t = time()
        EM.fit(rdd, n_clusters = K, max_iter = max_iter, criterion = None, verbose = False, tol=-1)
        time_spark[i,j] = time()-t

#np.save("time_nospark",time_nospark)
#np.save("time_spark",time_spark)

plt.figure()
plt.scatter(np.repeat(labels,sample_size), time_spark, c="r",label="Spark")
plt.scatter(np.repeat(labels,sample_size), time_nospark, c="b",label="No Spark")
plt.legend();

