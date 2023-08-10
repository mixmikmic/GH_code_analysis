import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

movies = spark.read.load("ml-latest-small/movies.csv", format='csv', header = True)
ratings = spark.read.load("ml-latest-small/ratings.csv", format='csv', header = True)
links = spark.read.load("ml-latest-small/links.csv", format='csv', header = True)
tags = spark.read.load("ml-latest-small/tags.csv", format='csv', header = True)

movies.show(5)

ratings.show(5)

links.show(5)

tags.show(5)

movies.groupby("genres").count().orderBy("count", ascending=False).show()

print 'Distinct values of ratings:'
print sorted(ratings.select('rating').distinct().rdd.map(lambda r: r[0]).collect())

tmp1 = ratings.groupBy("userID").count().select('count').rdd.min()[0]
tmp2 = ratings.groupBy("movieId").count().select('count').rdd.min()[0]
print 'For the users that rated movies and the movies that were rated:'
print 'Minimum number of ratings per user is {}'.format(tmp1)
print 'Minimum number of ratings per movie is {}'.format(tmp2)

tmp1 = ratings.groupBy("movieId").count().withColumnRenamed("count", "rating count").groupBy("rating count").count().orderBy('rating count').first()[1]
# Or use pandas: tmp1 = sum(ratings.groupBy("movieId").count().toPandas()['count'] == 1)
tmp2 = ratings.select('movieId').distinct().count()
print '{} out of {} movies are rated by only one user'.format(tmp1, tmp2)

print "Number of users:", ratings.select('userId').union(tags.select('userId')).distinct().count()

print "Number of users who rated movies:", ratings.select('userId').distinct().count()

print "Number of movies:", ratings.select('movieId').union(tags.select('movieId')).distinct().count()

print "Number of rated movies:", ratings.select('movieId').distinct().count()

ratings = ratings.select("userId", "movieId", "rating")

# inspect the schema of the data frame
ratings.printSchema()

# Below is a correct and efficient way to change the column types, but it somehow produced 
# errors on my laptop later when working on the dataframe due to some java issues...
# So I used the withColumn() method instead

"""
from pyspark.sql.types import *

schema = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("rating", FloatType(), True)])

df = spark.createDataFrame(ratings.rdd, schema)
"""

df = ratings.withColumn('userId', ratings['userId'].cast('int')).withColumn('movieId', ratings['movieId'].cast('int')).withColumn('rating', ratings['rating'].cast('float'))

# inspect the schema again
df.printSchema()

train, validation, test = df.randomSplit([0.6,0.2,0.2], seed = 0)

print "The number of ratings in each set: {}, {}, {}".format(train.count(), validation.count(), test.count())

mean_rating = train.groupby('movieId').mean().select('movieId','avg(rating)')
mean_rating = mean_rating.withColumnRenamed('avg(rating)','prediction')
mean_rating.show(5)

test.createOrReplaceTempView("test")
mean_rating.createOrReplaceTempView("mean_rating")

sqlDF = spark.sql("select test.*, mean_rating.prediction                    from test join mean_rating                    on test.movieId = mean_rating.movieId")
sqlDF.show(5)

# Define a function to calculate RMSE

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

def RMSE(predictions):
    return evaluator.evaluate(predictions)

print 'Using the mean rating of each movie as the prediction, the testing RMSE is ' + str(RMSE(sqlDF))

# Define a function to perform grid search and find the best ALS model
# based on the validation RMSE

from pyspark.ml.recommendation import ALS

def GridSearch(train, valid, num_iterations, reg_param, n_factors):
    min_rmse = float('inf')
    best_n = -1
    best_reg = 0
    best_model = None
    for n in n_factors:
        for reg in reg_param:
            als = ALS(rank = n, 
                      maxIter = num_iterations, 
                      seed = 0, 
                      regParam = reg,
                      userCol="userId", 
                      itemCol="movieId", 
                      ratingCol="rating", 
                      coldStartStrategy="drop")            
            model = als.fit(train)
            predictions = model.transform(valid)
            rmse = RMSE(predictions)     
            print '{} latent factors and regularization = {}: validation RMSE is {}'.format(n, reg, rmse)
            if rmse < min_rmse:
                min_rmse = rmse
                best_n = n
                best_reg = reg
                best_model = model
                
    pred = best_model.transform(train)
    train_rmse = RMSE(pred)
    print '\nThe best model has {} latent factors and regularization = {}:'.format(best_n, best_reg)
    print 'traning RMSE is {}; validation RMSE is {}'.format(train_rmse, min_rmse)
    return best_model

num_iterations = 10
ranks = [6, 8, 10, 12]
reg_params = [0.05, 0.1, 0.2, 0.4, 0.8]

start_time = time.time()
final_model = GridSearch(train, validation, num_iterations, reg_params, ranks)
print 'Total Runtime: {:.2f} seconds'.format(time.time() - start_time)

num_iterations = 15
ranks = [7, 8, 9]
reg_params = [0.1, 0.2, 0.3]

final_model = GridSearch(train, validation, num_iterations, reg_params, ranks)

pred_test = final_model.transform(test)
print 'The testing RMSE is ' + str(RMSE(pred_test))

pred_train = final_model.transform(train)
df = pred_train.toPandas()
x = np.arange(0.5, 5.1, 0.5) # To draw the red one-to-one line below

fig = plt.figure(figsize=(10,8))
plt.tick_params(labelsize=15)
plt.scatter(df.rating, df.prediction, s = 10, alpha = 0.2)
plt.plot(x, x, c = 'r')
plt.xlabel('rating', fontsize=20)
plt.ylabel('prediction', fontsize=20)
plt.title('Training Set', fontsize=20)
plt.show()

df = pred_test.toPandas()
fig = plt.figure(figsize=(10,8))
plt.tick_params(labelsize=15)
plt.scatter(df.rating, df.prediction, s = 10, alpha = 0.2)
plt.plot(x, x, c = 'r')
plt.xlabel('rating', fontsize=20)
plt.ylabel('prediction', fontsize=20)
plt.title('Testing Set', fontsize=20)
plt.show()



