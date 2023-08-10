from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
from pyspark.sql import Row, SQLContext
from pyspark.sql.functions import lit, udf
import pandas as pd
import numpy as np

df = pd.read_csv("purchases.csv", sep=";", names=["timestamp", "user_id", "product_id"], header= None)
df = df.drop("timestamp", axis=1)

rawdata = df.groupby("user_id").product_id.apply(set)

product_to_ix = {prod:i for i, prod in enumerate(df.product_id.unique())}
ix_to_product = {i:prod for i, prod in enumerate(df.product_id.unique())}

def expand_user(a):
    return [Rating(user, product_to_ix[item], 1) for user in a.index for item in a[user]]
ratings = expand_user(rawdata)

ratingsRDD = sc.parallelize(ratings)

training_RDD, test_RDD = ratingsRDD.randomSplit([8, 2], seed=123)
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

seed = 4242
iterations = 10
regularization_parameter =[i * 0.01 for i in range(1, 20, 2)]
ranks = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
errors = [[0]*len(regularization_parameter)] * len(ranks)

min_error = float('inf')
best_lambda = -1
best_lambda_index = -1
best_model = None
best_rank = -1
best_rank_index = -1


# Loop over all possible value fr lambda and rank to find the best parameters for our model that minimize the rmse
for i, rank in enumerate(ranks):
    for j, regParam in enumerate(regularization_parameter):
        model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations, lambda_=regParam)
        predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = np.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        errors[i][j] = error
        print('For lambda %s and rank %s the RMSE is %s' % (regParam, rank, error))
        if error < min_error:
            min_error = error
            best_lambda = regParam
            best_model = model
            best_rank = rank
            best_rank_index = i
            best_lambda_index = j
        with open('sparkLogging', 'a') as f:
            f.write("RMSE on testing set: {}, with rank: {}, lambda: {}\n".format(error, rank, regParam))


print('The best model was trained with lambda %s, rank %s and RMSE: %s' % (best_lambda, best_rank, min_error))

with open('sparkLoggingBest', 'a') as f:
    f.write("RMSE on testing set: {}, with rank: {} at index {}, lambda: {} at index {}\n".format(errors[best_rank_index][best_lambda_index], best_rank, best_lambda_index,  best_lambda, best_lambda_index))


seed = 4242
iterations = [5, 10, 15, 20]
regParam = 0.07
rank = 17

for iteration in iterations:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iteration, lambda_=0.07)
    predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = np.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    print('For %d iterations, the RMSE is %s' % (iteration, error))

model = ALS.train(ratingsRDD, rank=17, seed=4242, iterations=20, lambda_=0.07)

### replace users with list(df.user_id.unique()) to get recommendation for all users
users = [4, 1, 12, 11, 10]
for user in users:
    best_5_recommendations = model.recommendProducts(user, 5)
    print("_"*50)
    print("\nFor user %d we recommend the following 5 products:\n" %user)
    for recommendation in best_5_recommendations:
        print(" "*10, ix_to_product[recommendation.product])
        



