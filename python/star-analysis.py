import pyspark
sc = pyspark.SparkContext('local[*]')

from pyspark.sql import SQLContext
sql = SQLContext(sc)

janData = sql.read.json("../data/january.json")

janData.printSchema()

jan = janData.select(janData['actor'],janData['repo'],janData['type'])

jan = jan.filter(jan['type']=='WatchEvent')

jan.first()

from collections import namedtuple

User = namedtuple('User', 'name id')
Repo = namedtuple('Repo', 'name id')

def create_user(row):
    actor = row['actor']
    return User(actor.login,actor.id)

def create_repo(row):
    repo = row['repo']
    return Repo(repo.name,repo.id)

jan = jan.map(lambda row: (create_user(row),create_repo(row)))

jan.first()

from pyspark.mllib.recommendation import Rating 

def create_rating(user_repo):
    user = user_repo[0]
    repo = user_repo[1]
    return Rating(user.id, repo.id, 1)

ratings = jan.map(lambda user_repo: create_rating(user_repo))

ratings.first()

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
import os.path

rank = 5

model = None
model_path = '../models/january-implicit.mdl'
if os.path.exists(model_path):
    print('Loading model...')
    model = MatrixFactorizationModel.load(sc, model_path)
    print('Model loaded successfully!')
else:
    print('Training model...')
    model = ALS.trainImplicit(ratings, rank, seed=0)
    print('Model trained successfully!')

# TODO:: Evaluate the model.

if not os.path.exists(model_path):
    model.save(sc, model_path)

# Some helper RDDs/functions to find users and repos.
repos = jan.map(lambda x: x[1]).distinct().cache()
users = jan.map(lambda x: x[0]).distinct().cache()

def find_repo(name=None, id=None):
    if name is not None:
        return repos.filter(lambda repo: repo.name==name).first()
    if id is not None:
        return repos.filter(lambda repo: repo.id==id).first()

def find_user(name=None, id=None):
    if name is not None:
        return users.filter(lambda user: user.name==name).first()
    if id is not None:
        return users.filter(lambda user: user.id==id).first()

# Find the user 'nathanph'.
user = find_user('nathanph')

# Generate some recommendations for our user.
recommendations = model.recommendProducts(user.id, 5)
recommendations = list(map(lambda recommendation: (recommendation, find_repo(id=recommendation.product)), recommendations))

recommendations

# Find what repos the user has starred.
stars = jan.filter(lambda star: star[0].name==user.name)
stars = stars.map(lambda star: star[1]).collect()

stars

# Only display the recommended repos that the user has not already starred.
relevant_recommendations = list(filter(lambda recommendation: recommendation[1] not in stars, recommendations))
relevant_recommendations = list(map(lambda recommendation: recommendation[1], relevant_recommendations))
relevant_recommendations

# Print some hyperlinks.
_ = [print(url) for url in list(map(lambda repo: 'http://github.com/'+repo.name, relevant_recommendations))]



