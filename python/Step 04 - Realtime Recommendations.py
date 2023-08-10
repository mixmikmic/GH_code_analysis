from pyspark.mllib.recommendation import Rating

ratingsRDD = sc.textFile('ratings.dat')                .map(lambda l: l.split("::"))                .map(lambda p: Rating(
                                  user = int(p[0]), 
                                  product = int(p[1]),
                                  rating = float(p[2]), 
                                  )).cache()

(training, test) = ratingsRDD.randomSplit([0.8, 0.2])

numTraining = training.count()
numTest = test.count()

# verify row counts for each dataset
print("Total: {0}, Training: {1}, test: {2}".format(ratingsRDD.count(), numTraining, numTest))

from pyspark.mllib.recommendation import ALS

rank = 50
numIterations = 20
lambdaParam = 0.1
model = ALS.train(training, rank, numIterations, lambdaParam)

import numpy as np

pf = model.productFeatures().cache()

pf_keys = pf.sortByKey().keys().collect()
pf_vals = pf.sortByKey().map(lambda x: list(x[1])).collect()             
        
Vt = np.matrix(np.asarray(pf.values().collect()))

full_u = np.zeros(len(pf_keys))
full_u.itemset(1, 5) # user has rated product_id:1 = 5
recommendations = full_u*Vt*Vt.T

print("predicted rating value", np.sort(recommendations)[:,-10:])

top_ten_recommended_product_ids = np.where(recommendations >= np.sort(recommendations)[:,-10:].min())[1]

print("predict rating prod_id", np.array_repr(top_ten_recommended_product_ids))



