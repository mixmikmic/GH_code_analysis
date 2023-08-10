from pyspark.mllib.recommendation import Rating

new_user_ID = 0

new_user_ratings = [
     Rating(0,260,9),   # Star Wars (1977)
     Rating(0,1,8),     # Toy Story (1995)
     Rating(0,16,7),    # Casino (1995)
     Rating(0,25,8),    # Leaving Las Vegas (1995)
     Rating(0,32,9),    # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
     Rating(0,335,4),   # Flintstones, The (1994)
     Rating(0,379,3),   # Timecop (1994)
     Rating(0,296,7),   # Pulp Fiction (1994)
     Rating(0,858,10) , # Godfather, The (1972)
     Rating(0,50,8)     # Usual Suspects, The (1995)
    ]

new_user_ratings_RDD = sc.parallelize(new_user_ratings)

new_user_ratings_RDD.collect()

from pyspark.mllib.recommendation import Rating

ratings = sc.textFile('ratings.dat')                .map(lambda l: l.split("::"))                .map(lambda p: Rating(
                                  user = int(p[0]), 
                                  product = int(p[1]),
                                  rating = float(p[2]), 
                                  ))

ratings = ratings.union(new_user_ratings_RDD)

from pyspark.mllib.recommendation import ALS

rank = 50
numIterations = 20
lambdaParam = 0.1
model = ALS.train(ratings, rank, numIterations, lambdaParam)

# if there is an existing model, delete it
get_ipython().system('rm -rf ./recommender_model')

# save the model
model.save(sc, './recommender_model')

from pyspark.mllib.recommendation import MatrixFactorizationModel

model = MatrixFactorizationModel.load(sc, './recommender_model')

new_user_rated_movie_ids = map(lambda x: x[1], new_user_ratings)

# new_user_rated_movied_ids = [260, 1, 16, 25, 32, 335, 379, 296, 858, 50]

new_user_unrated_movies_RDD = ratings.filter(lambda r: r.product not in new_user_rated_movie_ids)                                      .map(lambda x: (new_user_ID, x[0]))                                      .distinct()

new_user_unrated_movies_RDD.take(5)

new_user_recommendations_RDD = model.predictAll(new_user_unrated_movies_RDD)

print(new_user_recommendations_RDD.take(10))

my_movie = sc.parallelize([(0, 500)]) # Quiz Show (1994)
individual_movie_rating_RDD = model.predictAll(my_movie)
individual_movie_rating_RDD.collect()



