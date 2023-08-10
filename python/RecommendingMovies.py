import conx

conx.download("http://files.grouplens.org/datasets/movielens/ml-1m.zip")

import pandas

# Read in the dataset, and do a little preprocessing,
# mostly to set the column datatypes.
users = pandas.read_csv('./ml-1m/users.dat', sep='::', 
                        engine='python', 
                        names=['userid', 'gender', 'age', 'occupation', 'zip']).set_index('userid')
ratings = pandas.read_csv('./ml-1m/ratings.dat', engine='python', 
                          sep='::', names=['userid', 'movieid', 'rating', 'timestamp'])
movies = pandas.read_csv('./ml-1m/movies.dat', engine='python',
                         sep='::', names=['movieid', 'title', 'genre']).set_index('movieid')
movies['genre'] = movies.genre.str.split('|')

users.age = users.age.astype('category')
users.gender = users.gender.astype('category')
users.occupation = users.occupation.astype('category')
ratings.movieid = ratings.movieid.astype('category')
ratings.userid = ratings.userid.astype('category')

n_movies = movies.shape[0]
n_users = users.shape[0]

n_movies, n_users

# Also, make vectors of all the movie ids and user ids. These are
# pandas categorical data, so they range from 1 to n_movies and 1 to n_users, respectively.
movie_ids = ratings.movieid.cat.codes.values
user_ids = ratings.userid.cat.codes.values
targets = ratings.rating.values

ds = conx.Dataset(name="Movie Recommendations", input_shapes=[(1,), (1,)])

pairs = []
for i in range(len(movie_ids)):
    ins = [movie_ids[i]], [user_ids[i]]
    targs = conx.onehot(targets[i] - 1, 5)
    pairs.append([ins, targs])

ds.load(pairs)
ds

net = conx.Network("Recommender")
net.add(conx.InputLayer("movie", 1))
net.add(conx.EmbeddingLayer("movie_embed", n_movies + 1, 32))
net.add(conx.FlattenLayer("movie_flatten", dropout=0.5))
net.add(conx.InputLayer("user", 1))
net.add(conx.EmbeddingLayer("user_embed", n_users + 1, 32))
net.add(conx.FlattenLayer("user_flatten", dropout=0.5))
net.add(conx.Layer("dense1", 128, activation="relu", dropout=0.5))
net.add(conx.BatchNormalizationLayer("batch norm1"))
net.add(conx.Layer("dense2", 128, activation="relu", dropout=0.5))
net.add(conx.BatchNormalizationLayer("batch norm2"))
net.add(conx.Layer("dense3", 128, activation="relu", dropout=0.5))
net.add(conx.Layer("output", 5, activation="softmax"))

net.connect("movie", "movie_embed")
net.connect("movie_embed", "movie_flatten")
net.connect("user", "user_embed")
net.connect("user_embed", "user_flatten")
net.connect("movie_flatten", "dense1")
net.connect("user_flatten", "dense1")
net.connect("dense1", "batch norm1")
net.connect("batch norm1", "dense2")
net.connect("dense2", "batch norm2")
net.connect("batch norm2", "dense3")
net.connect("dense3", "output")

net.compile(optimizer='adam', error='categorical_crossentropy')
net.dashboard()

net.propagate_to_image("output", [[1], [n_users]], visualize=True)

net.set_dataset(ds)

net.dataset.chop(0.01) ## retain 1%

net.dataset

net.dataset.split(.10) # save 10% for testing

net.dataset

net.train(epochs=20, plot=True)

net.propagate_to("movie_flatten", net.dataset.inputs[0][0])

net.train(epochs=5, plot=True)

net.propagate_to("movie_flatten", net.dataset.inputs[0][0])



