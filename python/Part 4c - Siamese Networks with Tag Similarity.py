import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

# visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from IPython.display import IFrame

# numeric and scientific processing
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from scipy.spatial.distance import dice, pdist, squareform

# misc
import os
import progressbar

# spotify API
import spotipy
import spotipy.util as util

# local caching
from joblib import Memory

# deep learning
from keras.models           import Model
from keras.layers           import Input, Lambda, Dense, Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.merge     import concatenate
from keras.optimizers       import Nadam
from keras import backend as K

# functions from Tutorial Part 1
import tutorial_functions as tut_func

SPOTIFY_USER = ""

os.environ["SPOTIPY_CLIENT_ID"]     = ""
os.environ["SPOTIPY_CLIENT_SECRET"] = ""

playlists = [
    
     {"name": "clubbeats",    "uri": "spotify:user:spotify:playlist:37i9dQZF1DXbX3zSzB4MO0"},
     {"name": "softpop",      "uri": "spotify:user:spotify:playlist:37i9dQZF1DWTwnEm1IYyoj"},
     {"name": "electropop",   "uri": "spotify:user:spotify:playlist:37i9dQZF1DX4uPi2roRUwU"},
     {"name": "rockclassics", "uri": "spotify:user:spotify:playlist:37i9dQZF1DWXRqgorJj26U"},
     {"name": "rockhymns",    "uri": "spotify:user:spotify:playlist:37i9dQZF1DX4vth7idTQch"},
     {"name": "soft_rock",    "uri": "spotify:user:spotify:playlist:37i9dQZF1DX6xOPeSOGone"},
     {"name": "metalcore",    "uri": "spotify:user:spotify:playlist:37i9dQZF1DWXIcbzpLauPS"}, 
     {"name": "metal",        "uri": "spotify:user:spotify:playlist:37i9dQZF1DWWOaP4H0w5b0"},
     {"name": "classic_metal","uri": "spotify:user:spotify:playlist:37i9dQZF1DX2LTcinqsO68"},
     {"name": "grunge",       "uri": "spotify:user:spotify:playlist:37i9dQZF1DX11ghcIxjcjE"},
     {"name": "hiphop",       "uri": "spotify:user:spotify:playlist:37i9dQZF1DWVdgXTbYm2r0"},
     {"name": "poppunk",      "uri": "spotify:user:spotify:playlist:37i9dQZF1DXa9wYJr1oMFq"},
     {"name": "classic",      "uri": "spotify:user:spotify:playlist:37i9dQZF1DXcN1fAVSf7CR"}
    
]

token = util.prompt_for_user_token(SPOTIFY_USER, 
                                   "playlist-modify-public", 
                                   redirect_uri="http://localhost/")

sp = spotipy.Spotify(auth=token)

memory = Memory(cachedir='/home/schindler/tmp/spotify/', verbose=0)

@memory.cache
def get_spotify_data(track_id):
    
    # meta-data
    track_metadata      = sp.track(track_id)
    album_metadata      = sp.album(track_metadata["album"]["id"])
    artist_metadata     = sp.artist(track_metadata["artists"][0]["id"])
    
    # feature-data
    sequential_features = sp.audio_analysis(track_id)
    trackbased_features = sp.audio_features([track_id])
    
    return track_metadata, album_metadata, artist_metadata, sequential_features, trackbased_features

# Get Playlist meta-data
playlists = tut_func.get_playlist_metadata(sp, playlists)

# Get track-ids of all playlist entries
playlists = tut_func.get_track_ids(sp, playlists)

num_tracks_total = np.sum([playlist["num_tracks"] for playlist in playlists])

# Fetch data and features from Spotify
pbar = progressbar.ProgressBar(max_value=num_tracks_total)
pbar.start()

raw_track_data      = []
processed_track_ids = []

for playlist in playlists:

    for track_id in playlist["track_ids"]:

        try:
            # avoid duplicates in the data-set
            if track_id not in processed_track_ids:

                # retrieve data from Spotify
                spotify_data = get_spotify_data(track_id)

                raw_track_data.append([playlist["name"], spotify_data])
                processed_track_ids.append(track_id)

        except Exception as e:
            print(e)

        pbar.update(len(raw_track_data))

# Aggregate Meta-data
metadata = tut_func.aggregate_metadata(raw_track_data)

# Aggregate Feature-data
feature_data = tut_func.aggregate_featuredata(raw_track_data, metadata)

# standardize sequential_features
feature_data -= feature_data.mean(axis=0)
feature_data /= feature_data.std(axis=0)

print("Track : " + " - ".join(metadata[["artist_name", "title"]].loc[330]) )
print("Labels: '" + "', '".join(metadata.loc[330].genres) + "'")

mlb = MultiLabelBinarizer()
mlb = mlb.fit(metadata.genres)

genres_bin = pd.DataFrame(mlb.transform(metadata.genres), columns=mlb.classes_)

# filter tracks without genre tags
genres_bin = genres_bin[genres_bin.sum(axis=1) > 0]

# copy the metadata
metadata_aligned = metadata

# create an additional column as an index into the feature-data
metadata_aligned["featurespace_index"] = metadata.index.values

# reduce the aligned metadata to those tracks with provided genre labels
metadata_aligned = metadata_aligned.iloc[genres_bin.index]

# reset the index
metadata_aligned = metadata_aligned.reset_index()

genres_bin.sum(axis=0).sort_values(ascending=False)[:50].plot(kind="bar", figsize=(16,6));

tf            = genres_bin.sum(axis=0)
idf           = np.log(genres_bin.shape[0] / tf)
genres_tf_idf = genres_bin / idf

# the pdist function calculates all pairwise distances
dists = pdist(genres_tf_idf, 'dice')

# pdist returns a list of results. We use the squareform function to convert this into a symmetric matrix
tag_based_track_similarities = pd.DataFrame((1 - squareform(dists)), 
                                            index   = genres_tf_idf.index,
                                            columns = genres_tf_idf.index)

query_track_idx = 33

print(metadata_aligned.iloc[query_track_idx])

# get the similar tracks from the distance-matrix
similar_tracks_idx = tag_based_track_similarities[query_track_idx].sort_values(ascending=False).index

# filter the aligned metadata
result        = metadata_aligned.loc[similar_tracks_idx]
result["sim"] = tag_based_track_similarities.loc[query_track_idx,
                                                 similar_tracks_idx].values

# show results
display_cols = ["artist_name", "title", "album_name", "year", "genres", "playlist", "sim"]
result[display_cols][:10]

def create_siamese_network(input_dim):

    # input layers
    input_left  = Input(shape=input_dim)
    input_right = Input(shape=input_dim)

    # shared fully connected layers
    shared_fc_1 = Dense(100, activation="selu")
    shared_fc_2 = Dense(100, activation="selu")    
    
    # siamese layers
    left_twin  = shared_fc_1(shared_fc_2(input_left))
    right_twin = shared_fc_1(shared_fc_2(input_right))

    # calc difference
    distance = Lambda(tut_func.euclidean_distance,
                      output_shape=lambda x: x[0])([left_twin, right_twin])

    return Model([input_left, input_right], distance)

def create_pairs_with_sims_and_identity(feature_data, metadata, num_pairs_per_track, sims):
    
    data_pairs = []
    labels     = []
    
    for row_id, q_track in metadata.sample(frac=1).iterrows():
        
        # identical pair
        data_pairs.append([feature_data[[row_id]][0], feature_data[[row_id]][0]])
        labels.append(1)
                
        for _ in range(num_pairs_per_track):
            
            # search similar and dissimilar examples
            pos_example = metadata[metadata.playlist == q_track.playlist].sample(1)
            neg_example = metadata[metadata.playlist != q_track.playlist].sample(1)

            # create feature pairs
            
            # similar pair
            data_pairs.append([feature_data[[metadata.loc[row_id].featurespace_index]][0], feature_data[[pos_example.featurespace_index]][0]])
            sim_val = sims.iloc[row_id, pos_example.index].values[0] - 0.1
            labels.append(np.max([0, sim_val]))
            
            # dissimilar pair
            data_pairs.append([feature_data[[metadata.loc[row_id].featurespace_index]][0], feature_data[[neg_example.featurespace_index]][0]])
            sim_val = sims.iloc[row_id, neg_example.index].values[0] - 0.1
            labels.append(np.max([0, sim_val]))
            

    return np.array(data_pairs), np.array(labels)

data_pairs, labels = create_pairs_with_sims_and_identity(feature_data, 
                                                         metadata_aligned, 
                                                         10, 
                                                         tag_based_track_similarities)

data_pairs.shape

# define the model
model = create_siamese_network(data_pairs[:,0].shape[1:])

# define the optimizer
opt = Nadam(lr=0.001)

# compile the model
model.compile(loss      = tut_func.contrastive_loss, 
              optimizer = opt)

model.fit([data_pairs[:, 0], data_pairs[:, 1]], 
                labels, 
                batch_size       = 24, 
                verbose          = 0, 
                epochs           = 25, 
                callbacks        = [tut_func.PlotLosses()], 
                validation_split = 0.1);

def similar(model, query_idx):
    
    print(metadata.iloc[query_idx])
    
    # calclulate predicted distances between query track and all others
    res = [model.predict([feature_data[[query_idx]], feature_data[[i]]]) for i in range(feature_data.shape[0])]

    # reshape
    res = np.array(res)
    res = res.reshape(res.shape[0])

    # get sorted indexes in ascending order (smallest distance to query track first)
    si = np.argsort(res)
    
    # output filter
    display_cols = ["artist_name", "title", "album_name", "year","playlist"]
    
    return metadata.loc[si, display_cols][:10]

similar(model, 33)

def aggregate_features_sequential(seq_data, track_data, len_segment_frames, len_segment_sec, m_data, with_year=False, with_popularity=False):
    
    # sequential data
    segments = seq_data["segments"]
    sl       = len(segments)
    
    mfcc              = np.array([s["timbre"]            for s in segments])
    chroma            = np.array([s["pitches"]           for s in segments])
    loudness_max      = np.array([s["loudness_max"]      for s in segments]).reshape((sl,1))
    loudness_start    = np.array([s["loudness_start"]    for s in segments]).reshape((sl,1))
    loudness_max_time = np.array([s["loudness_max_time"] for s in segments]).reshape((sl,1))
    duration          = np.array([s["duration"]          for s in segments]).reshape((sl,1))
    confidence        = np.array([s["confidence"]        for s in segments]).reshape((sl,1))
        
    # concatenate sequential features
    sequential_features = np.concatenate([mfcc, chroma, loudness_max, loudness_start, 
                                          loudness_max_time, duration, confidence], axis=1)

    # calculate length of segment (in ms)
    length_of_track        = track_data[0]["duration_ms"] / 1000.
    length_of_segment      = length_of_track / len(segments)
    num_segments_for_n_sek = int(np.round(len_segment_sec / length_of_segment))
    
    # select a random lstm-input-segment from the aggregated feature data
    offset  = np.random.randint(0, sl - num_segments_for_n_sek)
    segment = sequential_features[offset:(offset+num_segments_for_n_sek),:]
    
    # re-scale segment length to desired length (in seconds)
    x  = np.arange(segment.shape[0])
    y  = np.arange(segment.shape[1])
    xx = np.linspace(x.min(),x.max(),len_segment_frames)

    newKernel = RectBivariateSpline(x,y,segment, kx=2,ky=2)
    segment   = newKernel(xx,y)
        
    # track-based data
    track_features = [track_data[0]["acousticness"],     # acoustic or not?
                      track_data[0]["danceability"],     # danceable?
                      track_data[0]["energy"],           # energetic or calm?
                      track_data[0]["instrumentalness"], # is somebody singing?
                      track_data[0]["liveness"],         # live or studio?
                      track_data[0]["speechiness"],      # rap or singing?
                      track_data[0]["tempo"],            # slow or fast?
                      track_data[0]["time_signature"],   # 3/4, 4/4, 6/8, etc.
                      track_data[0]["valence"]]          # happy or sad?
    
    if with_year:
        track_features.append(int(m_data["year"]))
        
    if with_popularity:
        track_features.append(int(m_data["popularity"]))
        
    
    return segment, track_features

# number of input-vectors for the LSTM
len_segment_frames = 24

# number of seconds these vectors describe
len_segment_sec    = 6.

sequential_features = []
trackbased_features = []

for i, (_, spotify_data) in enumerate(raw_track_data):
    
    _, _, _, f_sequential, f_trackbased = spotify_data
    
    seq_feat, track_feat = aggregate_features_sequential(f_sequential, 
                                                         f_trackbased, 
                                                         len_segment_frames,
                                                         len_segment_sec,
                                                         metadata.loc[i],
                                                         with_year=True,
                                                         with_popularity=True)
    
    sequential_features.append(seq_feat)
    trackbased_features.append(track_feat)
    
sequential_features = np.asarray(sequential_features)
trackbased_features = np.asarray(trackbased_features)

print("sequential_features.shape:", sequential_features.shape)
print("trackbased_features.shape:", trackbased_features.shape)

# standardize sequential_features
rows, x, y = sequential_features.shape
sequential_features  = sequential_features.reshape(rows, (x * y))
sequential_features -= sequential_features.mean(axis=0)
sequential_features /= sequential_features.std(axis=0)
sequential_features  = sequential_features.reshape(rows, x, y)

# standardize trackbased_features
trackbased_features -= trackbased_features.mean(axis=0)
trackbased_features /= trackbased_features.std(axis=0)

def create_pairs_with_sims_and_identity_segments(sequential_features, trackbased_features, metadata, num_pairs_per_track, sims):
    
    data_pairs_seq   = []
    data_pairs_track = []
    labels           = []
    
    for row_id, q_track in metadata.sample(frac=1).iterrows():
        
        query_segment      = sequential_features[[metadata.loc[row_id].featurespace_index]][0]
        query_track_vector = trackbased_features[[metadata.loc[row_id].featurespace_index]][0]
        
        data_pairs_seq.append([query_segment, query_segment])
        data_pairs_track.append([query_track_vector, query_track_vector])
        labels.append(1)
        
        for _ in range(num_pairs_per_track):
            
            # search similar and dissimilar examples
            pos_example = metadata[metadata.playlist == q_track.playlist].sample(1)
            neg_example = metadata[metadata.playlist != q_track.playlist].sample(1)

            # create feature pairs
            data_pairs_seq.append([query_segment, sequential_features[[pos_example.featurespace_index]][0]])
            data_pairs_track.append([query_track_vector, trackbased_features[[pos_example.featurespace_index]][0]])
            labels.append(np.max([0, sims.iloc[row_id, pos_example.index].values[0] - 0.1]))

            data_pairs_seq.append([query_segment, sequential_features[[neg_example.featurespace_index]][0]])
            data_pairs_track.append([query_track_vector, trackbased_features[[neg_example.featurespace_index]][0]])
            labels.append(np.max([0, sims.iloc[row_id, neg_example.index].values[0] - 0.1]))

    return np.array(data_pairs_seq), np.array(data_pairs_track), np.asarray(labels)

data_pairs_seq, data_pairs_track, labels = create_pairs_with_sims_and_identity_segments(sequential_features,
                                                                                        trackbased_features,
                                                                                        metadata_aligned, 
                                                                                        10,
                                                                                        tag_based_track_similarities)

def create_siamese_network_with_lstm(data_pairs_seq, data_pairs_track):

    # sequential input
    input_seq_left  = Input(shape=data_pairs_seq[:, 0].shape[1:])
    input_seq_right = Input(shape=data_pairs_seq[:, 0].shape[1:])

    # track-based input
    input_track_left  = Input(shape=data_pairs_track[:, 0].shape[1:])
    input_track_right = Input(shape=data_pairs_track[:, 0].shape[1:])

    # shared bi-directional LSTM layer for the sequential features
    bdlstm = Bidirectional(LSTM(29, return_sequences=False, activation="selu"))

    # shared fully connected layers for the track-based features
    shared_fc_1 = Dense(11, activation="selu")
    shared_fc_2 = Dense(11, activation="selu")   

    # left twin
    seq_resp_left   = bdlstm(input_seq_left)
    track_resp_left = shared_fc_1(shared_fc_2(input_track_left))
    left_twin       = concatenate([seq_resp_left, track_resp_left], axis=1)

    # right twin
    seq_resp_right   = bdlstm(input_seq_right)
    track_resp_right = shared_fc_1(shared_fc_2(input_track_right))
    right_twin       = concatenate([seq_resp_right, track_resp_right], axis=1)

    # calc difference
    distance = Lambda(tut_func.euclidean_distance,
                      output_shape=lambda x: x[0])([left_twin, right_twin])

    return Model([input_seq_left, input_seq_right, input_track_left, input_track_right], distance)

# define the model
model_rnn = create_siamese_network_with_lstm(data_pairs_seq, data_pairs_track)

# define the optimizer
opt = Nadam(lr=0.001)

# compile the model
model_rnn.compile(loss      = tut_func.contrastive_loss, 
                  optimizer = opt)

model_rnn.fit([data_pairs_seq[:, 0],  data_pairs_seq[:, 1], 
               data_pairs_track[:,0], data_pairs_track[:,1]], 
              labels, 
              batch_size       = 24, 
              verbose          = 0, 
              epochs           = 25,
              callbacks        = [tut_func.PlotLosses()], 
              validation_split = 0.1);

def similar_rnn(model, query_idx):
    
    print(metadata.iloc[query_idx])
    
    # calclulate predicted distances between query track and all others
    res = [model.predict([sequential_features[[query_idx]], sequential_features[[i]], 
                          trackbased_features[[query_idx]], trackbased_features[[i]]]) \
           for i in range(feature_data.shape[0])]

    # reshape
    res = np.array(res)
    res = res.reshape(res.shape[0])

    # get sorted indexes in ascending order (smallest distance to query track first)
    si = np.argsort(res)
    
    # output filter
    display_cols = ["artist_name", "title", "album_name", "year","playlist"]
    
    return metadata.loc[si, display_cols][:10]

similar_rnn(model_rnn, 33)

