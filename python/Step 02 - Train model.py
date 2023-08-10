get_ipython().system('head -3 ratings.dat')
get_ipython().system('echo')
get_ipython().system('tail -3 ratings.dat')

from pyspark.mllib.recommendation import Rating

ratingsRDD = sc.textFile('ratings.dat')                .map(lambda l: l.split("::"))                .map(lambda p: Rating(
                                  user = int(p[0]), 
                                  product = int(p[1]),
                                  rating = float(p[2]), 
                                  )).cache()

ratingsRDD.toDF().describe().show()

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

# if there is an existing model, delete it
get_ipython().system('rm -rf ./recommender_model')

# save the model
model.save(sc, './recommender_model')

get_ipython().system('find ./recommender_model')

mean_rating = training.map(lambda x: x[2]).mean()

# we need to structure the (user, product (movie)) pairs with the mean rating
predictions = test.map(lambda r: ((r.user, r.product), mean_rating))

# predictions.take(2) looks like this:
# [((1, 1193), 3.581855855112805), ((1, 661), 3.581855855112805)]
# 
# I.e.
# [((user, movie), rating), ...]

ratesAndPreds = test.map(lambda r: ((r.user, r.product), r.rating))                     .join(predictions)
    
# ratesAndPreds.take(2) looks like this: 
# [((4520, 2502), (3.0, 3.581855855112805)), ((1320, 1230), (5.0, 3.581855855112805))]
#
# I.e.    
# [((user, movie), (actual_rating, predicted_rating)), ...]
# 
# The (user, product (movie)) tuple is the key and the value is another tuple
# representing the actual rating and the mean rating.


# subtract the mean rating from the actual rating, square the result and take the 
# mean of the results
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()

import math
RMSE = math.sqrt(MSE)

print("Mean Squared Error = " + str(MSE))
print("Root Mean Squared Error = " + str(RMSE))

# get a RDD that is a list of user + product (movie) pairs 
test_without_rating = test.map(lambda r: (r.user, r.product))


# pass the (user, product (movie)) pairs to the model to predict the values
predictions = model.predictAll(test_without_rating)                    .map(lambda r: ((r.user, r.product), r.rating))

# predictions.take(2) looks like this:
# [((4904, 2346), 4.138371476123861), ((4904, 2762), 5.076268198843158)]
# 
# I.e.
# [((user, movie), rating), ...]

# next we join the a RDD with the actual rating to the predicted rating
ratesAndPreds = test.map(lambda r: ((r.user, r.product), r.rating))                     .join(predictions)

# ratesAndPreds.take(2) looks like this:
# [((4520, 2502), (3.0, 3.948614784605795)), ((1320, 1230), (5.0, 4.648257851849921))]
#
# I.e.
# [((user, movie), (actual_rating, predicted_rating)), ...]
# 
# The (user, product (movie)) tuple is the key and the value is another tuple
# representing the actual rating and the mean rating.    
    

# subtract the mean rating from the actual rating, square the result and take the 
# mean of the results
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    
import math
RMSE = math.sqrt(MSE)
    
print("Mean Squared Error = " + str(MSE))
print("Root Mean Squared Error = " + str(RMSE))

from pyspark.sql.functions import min, max

user = predictions.map(lambda x: int(x[0][0])).cache()
movie = predictions.map(lambda x: int(x[0][1])).cache()
se = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).cache()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ratingsDF = ratingsRDD.toDF()

max_user = ratingsDF.select(max('user')).take(1)[0]['max(user)']
max_movie = ratingsDF.select(max('product')).take(1)[0]['max(product)']

width = 10
height = 10
plt.figure(figsize=(width, height))
plt.xlim([0,max_user])
plt.ylim([0,max_movie])
plt.xlabel('User ID')
plt.ylabel('Movie ID')
plt.title('Prediction Squared Error')

ax = plt.gca()
ax.patch.set_facecolor('#898787') # dark grey background

colors = plt.cm.YlOrRd(se.collect())

plt.scatter(
    user.collect(), 
    movie.collect(), 
    s=1,
    edgecolor=colors)

plt.legend(
    title='Normalised Squared Error',
    loc="upper left", 
    bbox_to_anchor=(1,1),
    handles=[
        mpatches.Patch(color=plt.cm.YlOrRd(0),    label='0'),
        mpatches.Patch(color=plt.cm.YlOrRd(0.25), label='0.25'),
        mpatches.Patch(color=plt.cm.YlOrRd(0.5),  label='0.5'),
        mpatches.Patch(color=plt.cm.YlOrRd(0.75), label='0.75'),
        mpatches.Patch(color=plt.cm.YlOrRd(0.99), label='1')    
    ])

plt.show()



