import findspark
findspark.init('Path_to_Spark_Installation_Folder')

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

data = spark.read.csv('movie_ratings.csv', header = True, inferSchema = True)
data.printSchema()

data.describe().show()

# Splitting the data into train set and test set
train, test = data.randomSplit([0.8, 0.2])

# Developing recommnedation system model
# Alternative least square(ALS) method
from pyspark.ml.recommendation import ALS
als = ALS(maxIter = 5, #  number of iterations to run
          regParam = 0.01, # regularization parameter
          userCol = 'userId',
          itemCol = 'movieId',
          ratingCol = 'rating',
#           coldStartStrategy = 'drop' # we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
         )

# Fitting the model with training data
model = als.fit(train)

# Checking the prediction with test data
pred = model.transform(test)
pred.show(10)

# Evaluating the model
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(metricName = 'rmse', labelCol = 'rating', predictionCol = 'prediction')
rmse = evaluator.evaluate(pred.na.drop())
print('RMSE: %.2f'%rmse)

# How can we use this model to recommend a movie to a new single user
single_user = test.filter(test['userId'] == 11).select(['movieId', 'userId'])
single_user.show(10)

# Let's predict how this user going to like the above mentioned movies
recommendations = model.transform(single_user)
recommendations.orderBy('movieId', ascending = False).show(10)

# Let's check our prediction against the actual data to see how well our model perform
test.filter(test['userId'] == 11).orderBy('movieId', ascending = False).show(10)

