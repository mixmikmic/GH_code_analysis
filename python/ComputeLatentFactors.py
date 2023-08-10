import pandas as pd
from scipy.sparse import csr_matrix
import pickle

# Give a unique index for each user
users = pd.read_csv('./data/train_triplets.txt',sep='\t', header=None, usecols=[0], names=['user']).user.unique()
users = {u:i for (i,u) in enumerate(users)}
with open('./data/users.pkl', mode='wb') as f:
    pickle.dump(users, f)

# Give a unique index for each song
songs = pd.read_csv('./data/train_triplets.txt',sep='\t', header=None, usecols=[1], names=['song']).song.unique()
songs = {u:i for (i,u) in enumerate(songs)}
with open('./data/songs.pkl', mode='wb') as f:
    pickle.dump(songs, f)

# Load the users and song dictionaries
with open('./data/users.pkl', mode='rb') as f:
    users = pickle.load(f)
with open('./data/songs.pkl', mode='rb') as f:
    songs = pickle.load(f)

# Iterate over the triplets file, and save the listening data in row,col,data lists
row = []
col = []
data = []
with open('./data/train_triplets.txt') as f:
    for line in f:
        s = line.strip().split('\t')
        row.append(users[s[0]])
        col.append(songs[s[1]])
        data.append(int(s[2]))

# Create the sparse matrix
echonest_data = csr_matrix((data, (row, col)), shape=(len(users), len(songs)))
with open('./data/echonest_data.pkl', mode='wb') as f:
    pickle.dump(echonest_data, f)

import pickle
import wmf

# Load the data
with open('./data/echonest_data.pkl', mode='rb') as f:
    data = pickle.load(f)
# data = pd.read_pickle("data/test_matrix.pkl")

# Calcualte latent factors
S = wmf.log_surplus_confidence_matrix(data, alpha=2.0, epsilon=1e-6)
U, V = wmf.factorize(S, num_factors=40, lambda_reg=1e-5, num_iterations=10, init_std=0.01, verbose=True, dtype='float32', recompute_factors=wmf.recompute_factors_bias)

V.shape

with open('./data/users.pkl', mode='rb') as f:
    users = pickle.load(f)
with open('./data/songs.pkl', mode='rb') as f:
    songs = pickle.load(f)

U = {i:v for (i,v) in enumerate(U)}
V = {i:v for (i,v) in enumerate(V)}

users = {u:U[i] for (u,i) in users.items()}
songs = {s:V[i] for (s,i) in songs.items()}

# Save the song latent factors
with open('./data/user_latent_factors.pkl', mode='wb') as f:
    pickle.dump(users, f)
with open('./data/song_latent_factors.pkl', mode='wb') as f:
    pickle.dump(songs, f)

