import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx
import tensorflow as tf
import pymc3 as pm
import GPflow

plt.style.use('fivethirtyeight')


from Bio import SeqIO
from copy import deepcopy
from custom import load_model
from datetime import datetime
from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model, model_from_json
from keras import backend as K
from keras import objectives
from keras.callbacks import EarlyStopping
from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from random import sample

from Levenshtein import distance as levD
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from pymc3 import gp
from sklearn.model_selection import train_test_split
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean as eucD

from custom import encode_array, save_model, get_density_interval
from utils.data import load_sequence_and_metadata

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

# Let's start by loading the protein sequence FASTA file and metadata.
sequences, metadata = load_sequence_and_metadata(kind='csv')
assert len(sequences) == len(metadata)

# Filter for just human sequences, then split into training and test set.
metadata = metadata[metadata['Host Species'] == 'IRD:Human']
training_metadata = metadata[metadata['Collection Date'] < datetime(2017, 1, 1)]
training_idxs = [i for i, s in enumerate(sequences) if s.id in training_metadata['Sequence Accession'].values]

test_metadata = metadata[metadata['Collection Date'] >= datetime(2017, 1, 1)]
test_idxs = [i for i, s in enumerate(sequences) if s.id in test_metadata['Sequence Accession'].values]

# Encode as array.
sequence_array = encode_array(sequences)
training_array = sequence_array[training_idxs]
test_array = sequence_array[test_idxs]

training_sequences = [sequences[i] for i in training_idxs]
test_sequences = [sequences[i] for i in test_idxs]

test_array   # the one-of-K encoding of sequences.

training_array  # also a one-of-K encoding of sequences

# Sanity checks
assert len(training_array) == len(training_metadata)
assert len(test_array) == len(test_metadata)

# Diagnostic prints that may be helpful later
print(training_array.shape)

# # Set up VAE.
# with tf.device('/gpu:0'):
#     intermediate_dim = 1000
#     encoding_dim = 3
#     latent_dim = encoding_dim
#     epsilon_std = 1.0
#     nb_epoch = 250

#     x = Input(shape=(training_array.shape[1],))
#     z_mean = Dense(latent_dim)(x)
#     z_log_var = Dense(latent_dim)(x)

#     def sampling(args):
#         z_mean, z_log_var = args
#         epsilon = K.random_normal(shape=(latent_dim, ), mean=0.,
#                                   stddev=epsilon_std)
#         return z_mean + K.exp(z_log_var / 2) * epsilon


#     def vae_loss(x, x_decoded_mean):
#         xent_loss = training_array.shape[1] * objectives.binary_crossentropy(x, x_decoded_mean)
#         kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#         return xent_loss + kl_loss

#     z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
#     x_decoded_mean = Dense(training_array.shape[1], activation='sigmoid')(z_mean)

#     vae = Model(x, x_decoded_mean)
#     vae.compile(optimizer='adam', loss=vae_loss)

#     # build a model to project inputs on the latent space
#     encoder = Model(x, z_mean)
#     encoder_var = Model(x, z_log_var)

#     x_train, x_test = train_test_split(training_array)

#     early_stopping = EarlyStopping(monitor="val_loss", patience=2)


#     # build the decoder
#     encoded_input = Input(shape=(encoding_dim,))
#     # retrieve the last layer of the autoencoder model
#     decoder_layer = vae.layers[-1]
#     # create the decoder model
#     decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))


#     # Train the VAE to learn weights
#     vae.fit(x_train, x_train,
#             shuffle=True,
#             epochs=nb_epoch,
#             validation_data=(x_test, x_test),
#             callbacks=[early_stopping],
#            )

# save_model(vae, 'trained_models/vae')
# save_model(encoder, 'trained_models/encoder')
# save_model(decoder, 'trained_models/decoder')

# with open('trained_models/vae.yaml', 'r+') as f:
#     yaml_spec = f.read()

vae = load_model('trained_models/vae')
encoder = load_model('trained_models/encoder')
decoder = load_model('trained_models/decoder')

training_embeddings_mean = encoder.predict(training_array)
training_embeddings_mean.shape

test_embeddings_mean = encoder.predict(test_array)
test_embeddings_hull = ConvexHull(test_embeddings_mean)

pd.DataFrame(test_embeddings_mean)

lowp, highp = get_density_interval(99, training_embeddings_mean, axis=0)
lowp, highp

# lowp, highp = get_density_interval(97.5, test_embeddings_mean, axis=0)
# lowp, highp

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(training_embeddings_mean[:, 0], 
           training_embeddings_mean[:, 1], 
           training_embeddings_mean[:, 2], 
           c=training_metadata['Collection Date'], cmap='viridis')
# Plot the convex hull of the 2017 sequences.
for simplex in test_embeddings_hull.simplices:
    ax.plot(test_embeddings_mean[simplex, 0], 
            test_embeddings_mean[simplex, 1], 
            test_embeddings_mean[simplex, 2], 'r--', lw=0.5)
ax.set_xlim(lowp[0], highp[0])
ax.set_ylim(lowp[1], highp[1])
ax.set_zlim(lowp[2], highp[2])
ax.set_facecolor('white')
plt.show()

training_metadata.head()

training_metadata.loc[:, 'coords0'] = training_embeddings_mean[:, 0]
training_metadata.loc[:, 'coords1'] = training_embeddings_mean[:, 1]
training_metadata.loc[:, 'coords2'] = training_embeddings_mean[:, 2]
training_metadata.tail()
# training_metadata.to_csv('data/metadata_with_embeddings.csv')

# Generate pairs of random indices.
indices = np.random.randint(low=0, high=len(training_array), size=(1000, 2))
indices

lev_dists = []
euc_dists = []
for i1, i2 in indices:
    lev_dist = levD(str(training_sequences[i1].seq), str(training_sequences[i2].seq))
    euc_dist = eucD(training_embeddings_mean[i1], training_embeddings_mean[i2])
    
    lev_dists.append(lev_dist)
    euc_dists.append(euc_dist)
    
fig = plt.figure(figsize=(5,5))
plt.scatter(lev_dists, euc_dists, alpha=0.1)
plt.xlabel('Levenshtein Distance')
plt.ylabel('Euclidean Distance')

from ipywidgets import interact
from ipywidgets.widgets import Dropdown

tm_coords = deepcopy(training_metadata)  # tm_coords means "training metadata with coordinates"
tm_coords['coord0'] = training_embeddings_mean[:, 0]
tm_coords['coord1'] = training_embeddings_mean[:, 1]
tm_coords['coord2'] = training_embeddings_mean[:, 2]
avg_coords_by_day = tm_coords.groupby('Collection Date')                            [['coord0', 'coord1', 'coord2']].median()                            .resample("D").median().dropna().reset_index()

avg_coords_by_quarter = tm_coords.groupby('Collection Date')                            [['coord0', 'coord1', 'coord2']].median().                            resample("Q").median().dropna().reset_index()

        
avg_coords_by_week = tm_coords.groupby('Collection Date')                            [['coord0', 'coord1', 'coord2']].median().                            resample("W").median().dropna().reset_index()

@interact(elev=(-180, 180, 10), azim=(0, 360, 10))
def plot_daily_avg(elev, azim):

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax1.scatter(xs=avg_coords_by_week['coord0'],
                ys=avg_coords_by_week['coord1'],
                zs=avg_coords_by_week['coord2'],
                c=avg_coords_by_week['Collection Date'], 
                cmap='viridis')
    ax1.view_init(elev, azim)
    ax1.set_title('by day')
    
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    ax2.scatter(xs=avg_coords_by_quarter['coord0'], 
                ys=avg_coords_by_quarter['coord1'], 
                zs=avg_coords_by_quarter['coord2'], 
                c=avg_coords_by_quarter['Collection Date'], 
                cmap='viridis')
    ax2.view_init(elev=elev, azim=azim)
    ax2.set_title('by quarter')
    
    ax1.set_xlim(ax2.get_xlim())
    ax1.set_ylim(ax2.get_ylim())
    ax1.set_zlim(ax2.get_zlim())
    
    for simplex in test_embeddings_hull.simplices:
        ax1.plot(test_embeddings_mean[simplex, 0], 
                 test_embeddings_mean[simplex, 1], 
                 test_embeddings_mean[simplex, 2], 
                 'k--', lw=0.2)
        
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    plt.show()

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,3,1)
ax1.plot(avg_coords_by_quarter['coord0'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Coordinate Value')
ax1.set_title('Coordinate 0')

ax2 = fig.add_subplot(1,3,2)
ax2.plot(avg_coords_by_quarter['coord1'])
ax2.set_xlabel('Time')
ax2.set_ylabel('Coordinate Value')
ax2.set_title('Coordinate 1')

ax3 = fig.add_subplot(1,3,3)
ax3.plot(avg_coords_by_quarter['coord2'])
ax3.set_xlabel('Time')
ax3.set_ylabel('Coordinate Value')
ax3.set_title('Coordinate 2')
plt.show()

import theano.tensor as tt
x_vals = np.array(range(len(avg_coords_by_quarter))).reshape(-1, 1).astype('float32')

def build_coords_model(coordinate):
    y_vals = avg_coords_by_quarter[coordinate].values.astype('float32')
    
    print(x_vals.shape, y_vals.shape)

    with pm.Model() as model:
        # l = pm.HalfCauchy('l', beta=20)
        l = pm.Uniform('l', 0, 30)

        # Covariance function
        log_s2_f = pm.Uniform('log_s2_f', lower=-10, upper=5)
        s2_f = pm.Deterministic('s2_f', tt.exp(log_s2_f))
        # s2_f = pm.HalfCauchy('s2_f', beta=2)
        # f_cov = s2_f * pm.gp.cov.Matern52(input_dim=1, lengthscales=l)
        f_cov = s2_f * pm.gp.cov.ExpQuad(input_dim=1, lengthscales=l)

        # Sigma
        log_s2_n = pm.Uniform('log_s2_n', lower=-10, upper=5)
        s2_n = pm.Deterministic('s2_n', tt.exp(log_s2_n))
        # s2_n = pm.HalfCauchy('s2_n', beta=2)

        y_obs = pm.gp.GP('y_obs', cov_func=f_cov, sigma=s2_n, 
                         observed={'X':x_vals, 
                                   'Y':y_vals})
        trace = pm.sample(2000)

        pp_x = np.arange(len(avg_coords_by_quarter)+2)[:, None]
        gp_samples = pm.gp.sample_gp(trace=trace, gp=y_obs, X_values=pp_x, samples=1000)

    return gp_samples

coord0_preds = build_coords_model('coord0')

coord1_preds = build_coords_model('coord1')

coord2_preds = build_coords_model('coord2')

coord0_preds.shape, coord1_preds.shape, coord2_preds.shape, 

from random import sample
def plot_coords_with_groundtruth(coord_preds, data, ax):
    pp_x = np.arange(len(avg_coords_by_quarter)+2)[:, None]
    for x in coord_preds:
        ax.plot(pp_x, x, color='purple', alpha=0.1)
    ax.plot(x_vals, data, 'oy', ms=10);
    ax.set_xlabel("quarter index");
    ax.set_ylabel("coordinate");
    ax.set_title("coordinate posterior predictive distribution");
    
    return ax

fig = plt.figure(figsize=(15,5))
ax0 = fig.add_subplot(1,3,1)
ax1 = fig.add_subplot(1,3,2)
ax2 = fig.add_subplot(1,3,3)
plot_coords_with_groundtruth(coord0_preds, avg_coords_by_quarter['coord0'], ax0)
plot_coords_with_groundtruth(coord1_preds, avg_coords_by_quarter['coord1'], ax1)
plot_coords_with_groundtruth(coord2_preds, avg_coords_by_quarter['coord2'], ax2)

test_coords_embed = deepcopy(test_metadata)
test_coords_embed['coord0'] = test_embeddings_mean[:, 0]
test_coords_embed['coord1'] = test_embeddings_mean[:, 1]
test_coords_embed['coord2'] = test_embeddings_mean[:, 2]
test_coords_embed = test_coords_embed.set_index('Collection Date').resample('Q').mean().reset_index()
test_coords_embed.to_csv('data/test_metadata_with_embeddings.csv')
test_coords_embed

# test_coords_embed[['coord0', 'coord1', 'coord2']].values
decoder.predict(test_coords_embed[['coord0', 'coord1', 'coord2']].values)

from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper
from bokeh.layouts import row

def ecdf(data):
    x, y = np.sort(data), np.arange(1, len(data)+1) / len(data)
    return x, y

def percentile_xy(x, y, perc):
    return np.percentile(x, perc), np.percentile(y, perc)


def plot_forecast_hpd(coord_preds, coords_test, c, quarter):
    
    
    x, y = ecdf(coord_preds[:, -3+quarter])
    y_perc = (y * 100).astype('int')
    df = pd.DataFrame(dict(x=x, y=y, y_perc=y_perc))
    
    df['color'] = df['y_perc'].apply(lambda x: 'red' if x > 2.5 and x < 97.5 else 'blue')
    src = ColumnDataSource(df)

    
    # Instantiate Figure
    p = figure(plot_width=300, plot_height=300, 
               x_axis_label="coordinate value", y_axis_label="cumulative distribution",
               title="")

    # Make scatter plot.
    p.scatter('x', 'y', 
              color='color',
              source=src)
    p.ray(x=[coords_test['coord{0}'.format(c)][quarter-1]], y=[0], angle=np.pi / 2, length=0, line_width=2, color='blue')
#     p.scatter(*percentile_xy(x, y, 2.5), color='red')
#     p.scatter(*percentile_xy(x, y, 97.5), color='red')

    # Add Hover tool
    hovertool = HoverTool()
    hovertool.tooltips = [
        ("Value", "@x"),
        ("Percentile", "@y_perc")
    ]
    p.add_tools(hovertool)

    return p

q = 2
p0 = plot_forecast_hpd(coord0_preds, test_coords_embed, 0, quarter=q)
p1 = plot_forecast_hpd(coord1_preds, test_coords_embed, 1, quarter=q)
p2 = plot_forecast_hpd(coord2_preds, test_coords_embed, 2, quarter=q)

show(row(p0, p1, p2))

# coord0_preds[sample(range(len(coord0_preds)), 100)].tolist()

output_notebook()

p = figure(plot_width=300, plot_height=280, x_axis_label='Calendar Quarter', y_axis_label='Coordinate Value')
pp_x = np.arange(len(avg_coords_by_quarter)+2)[:, None]

coord = 2
coord_pred_dict = {'coord0': coord0_preds,
                   'coord1': coord1_preds,
                   'coord2': coord2_preds,}

n_samples = 200
xs = list(pp_x.reshape(-1).repeat(n_samples).reshape(-1, n_samples).T.tolist())
ys = coord_pred_dict['coord{0}'.format(coord)][sample(range(len(coord_pred_dict['coord{0}'.format(coord)])), n_samples)].tolist()
p.multi_line(xs=xs, 
             ys=ys, 
             color='#fff176', 
             alpha=1)

p.circle(x_vals.reshape(-1), avg_coords_by_quarter['coord{0}'.format(coord)], color='#616161')
p.circle(pp_x[-2:].reshape(-1), test_coords_embed['coord{0}'.format(coord)], color='#ff5252', radius=1)
show(p)

one_quarter_pred = np.array([coord0_preds[:, -2], coord1_preds[:, -2], coord2_preds[:, -2]]).T
two_quarter_pred = np.array([coord0_preds[:, -1], coord1_preds[:, -1], coord2_preds[:, -1]]).T

# Grab out the 95% HPD interval from the data, filter out the data that are outside of HPD
oneQ_low, oneQ_high = get_density_interval(95, one_quarter_pred)
oneQ_low, oneQ_high

one_quarter_pred

test_embeddings_mean

oneQ_hull = ConvexHull(one_quarter_pred)
twoQ_hull = ConvexHull(two_quarter_pred)

oneQ_hull.vertices

dim = 2
one_quarter_pred[oneQ_hull.vertices][:, dim].min(), one_quarter_pred[oneQ_hull.vertices][:, dim].max()
# two_quarter_pred[twoQ_hull.vertices][:, dim].min(), two_quarter_pred[twoQ_hull.vertices][:, dim].max()

from ipywidgets import IntSlider
# @interact(elev=range(-180, 180, 20), azim=range(0, 360, 20))
def plot_predicted_coords(elev, azim):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter(test_coords_embed['coord0'], test_coords_embed['coord1'], 
               test_coords_embed['coord2'], s=200)
    # ax.scatter(one_quarter_pred[:, 0], one_quarter_pred[:, 1], one_quarter_pred[:, 2], alpha=0.1)
    # ax.scatter(two_quarter_pred[:, 0], two_quarter_pred[:, 1], two_quarter_pred[:, 2], alpha=0.1)
    ax.set_facecolor('white')
    ax.view_init(elev=elev, azim=azim)
    
    for simplex in oneQ_hull.simplices:
        ax.plot(one_quarter_pred[simplex, 0], 
                 one_quarter_pred[simplex, 1], 
                 one_quarter_pred[simplex, 2], 
                 'b-', lw=0.2)
    plt.show()
    
elev_slider = IntSlider(value=0, min=-180, max=180, step=20)
azim_slider = IntSlider(value=60, min=0, max=360, step=20)
interact(plot_predicted_coords, elev=elev_slider, azim=azim_slider)

from custom import embedding2binary, binary2chararray
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from collections import Counter

def embedding2seqs(decoder, preds, sequences):
    binary = embedding2binary(decoder, preds)
    chararray = binary2chararray(sequences, binary)
    strs = [''.join(i for i in k) for k in chararray]
    strcounts = Counter(strs)
    seqrecords = []
    for s, c in sorted(strcounts.items(), key=lambda x: x[1], reverse=True):
        s = Seq(s)
        sr = SeqRecord(s, id='weight_' + str(c))
        seqrecords.append(sr)
        
    return binary, chararray, strcounts, strs, seqrecords
    
oneQ_binary, oneQ_chararray, oneQ_strcounts, oneQ_strs, oneQ_seqrecords = embedding2seqs(decoder, one_quarter_pred, sequences)
twoQ_binary, twoQ_chararray, twoQ_strcounts, twoQ_strs, twoQ_seqrecords = embedding2seqs(decoder, two_quarter_pred, sequences)

twoQ_strcounts.keys()

# oneQ_binary = embedding2binary(decoder, one_quarter_pred)
# oneQ_chararray = binary2chararray(sequences, oneQ_binary)

# twoQ_binary = embedding2binary(decoder, two_quarter_pred)
# twoQ_chararray = binary2chararray(sequences, twoQ_binary)


# # Data structure here: 
# oneQ_strs = Counter([''.join(i for i in k) for k in oneQ_chararray])

# oneQ_seqrecords = []
# for s, c in sorted(oneQ_strs.items(), key=lambda x: x[1], reverse=True):
#     s = Seq(s)
#     sr = SeqRecord(s, id='weight_' + str(c))
#     oneQ_seqrecords.append(sr)
    
# # Write to disk
SeqIO.write(oneQ_seqrecords, 'data/oneQ_predictions.fasta', 'fasta')
SeqIO.write(twoQ_seqrecords, 'data/twoQ_predictions.fasta', 'fasta')

test_binarray = embedding2binary(decoder, test_coords_embed.loc[:, ['coord0', 'coord1', 'coord2']].values)
test_chararray = binary2chararray(sequences, test_binarray)
test_strs = [''.join(i for i in k) for k in test_chararray]
test_seqrecords = [SeqRecord(seq=Seq(i)) for n, i in enumerate(test_strs)]

twoQ_strcounts[test_strs[1]]
oneQ_strcounts[test_strs[0]]

sorted(oneQ_strcounts.values(), reverse=True)[0:3]

sorted(twoQ_strcounts.values(), reverse=True)[0:3]

from palettable.tableau import Tableau_20
from matplotlib.colors import ListedColormap
from itertools import cycle
from datetime import datetime

def make_predcoords_df(strcounts, strs, chararray, pred):
    """
    Just a utility function for the notebook. DO NOT PUT INTO PRODUCTION!
    """

    # cm = discrete_cmap(len(strcounts.keys()), base_cmap=Tableau_20)
    cmap_mpl = dict()
    for (seq, rgb) in zip(strcounts.keys(), cycle(Tableau_20.mpl_colors)):
        cmap_mpl[seq] = rgb

    cmap_hexcol = dict()
    for (seq, hexcol) in zip(strcounts.keys(), cycle(Tableau_20.hex_colors)):
        cmap_hexcol[seq] = hexcol
    
    mpl_colors = [cmap_mpl[s] for s in strs]
    hex_colors = [cmap_hexcol[s] for s in strs]    
    
    predcoords = pd.DataFrame()
    predcoords['coords0'] = pred[:, 0]
    predcoords['coords1'] = pred[:, 1]
    predcoords['coords2'] = pred[:, 2]
    predcoords['matplotlib_colors'] = mpl_colors
    predcoords['hexdecimal_colors'] = hex_colors
    
    return predcoords
    
oneQ_predcoords = make_predcoords_df(oneQ_strcounts, oneQ_strs, oneQ_chararray, one_quarter_pred)
twoQ_predcoords = make_predcoords_df(twoQ_strcounts, twoQ_strs, twoQ_chararray, two_quarter_pred)

oneQ_predcoords.to_csv('data/oneQ_prediction_coords_with_colors.csv')
twoQ_predcoords.to_csv('data/twoQ_prediction_coords_with_colors.csv')

dim1 = 'coords0'
dim2 = 'coords2'

def make_bokeh_plot_predcoord_bounding_boxes(predcoords, dim1, dim2):
    """
    This is just a plot made for diagnostic purposes. DO NOT PUT INTO PRODUCTION!
    """
    
    assert dim1 != dim2
    p = figure(plot_width=400, plot_height=300)
    xs_all = []
    ys_all = []
    colors = []
    starttime = datetime.now()
    for (mpl_color, hex_color), dat in oneQ_predcoords.groupby(['matplotlib_colors', 'hexdecimal_colors']):
        d = dat[[dim1, dim2]]
        
        if len(d) >=10:
            xs = []
            ys = []
            hull = ConvexHull(d[[dim1, dim2]])
            for v in hull.vertices:
                xs.append(d.iloc[v][dim1])
                ys.append(d.iloc[v][dim2])
            xs.append(xs[0])
            ys.append(ys[0])
            xs_all.append(xs)
            ys_all.append(ys)
            colors.append(hex_color)
    p.multi_line(xs_all, ys_all, color=colors)
    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f'elapsed time: {elapsed} s')

    show(p)
    
make_bokeh_plot_predcoord_bounding_boxes(twoQ_predcoords, 'coords0', 'coords2')

most_probable_seq = 'MKTIIALSYILCLVFAQKIPGNDNSTATLCLGHHAVPNGTIVKTITNDRIEVTNATELVQNSSIGEICDSPHQILDGENCTLIDALLGDPQCDGFQNKKWDLFVERSKAYSNCYPYDVPDYASLRSLVASSGTLEFNNESFNWTGVTQNGTSSACIRRSSSSFFSRLNWLTHLNYTYPALNVTMPNNEQFDKLYIWGVHHPGTDKDQIFLYAQSSGRITVSTKRSQQAVIPNIGSRPRIRDIPSRISIYWTIVKPGDILLINSTGNLIAPRGYFKIRSGKSSIMRSDAPIGKCKSECITPNGSIPNDKPFQNVNRITYGACPRYVKHSTLKLATGMRNVPEKQTRGIFGAIAGFIENGWEGMVDGWYGFRHQNSEGRGQAADLKSTQAAIDQINGKLNRLIGKTNEKFHQIEKEFSEVEGRIQDLEKYVEDTKIDLWSYNAELLVALENQHTIDLTDSEMNKLFEKTKKQLRENAEDMGNGCFKIYHKCDNACIGSIRNGTYDHNVYRDEALNNRFQIKGVELKSGYKDWILWISFAISCFLLCVALLGFIMWACQKGNIRCNICIAAAA'

from Levenshtein import distance
levDs_from_preds = [distance(str(record.seq), most_probable_seq) for record in test_sequences]

def ecdf_scatter(data):
    x, y = np.sort(data), np.arange(1, len(data)+1) / len(data)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x, y)
    
    return ax
    
    
ax = ecdf_scatter(levDs_from_preds)
ax.set_xlabel('Number of amino acids differing')
ax.set_ylabel('Cumulative distribution')

import yaml

with open('data/vaccine_strains.yaml', 'r+') as f:
    vaccine_strains = yaml.load(f)
    
set(vaccine_strains.values())

training_metadata['Strain Name'] = training_metadata['Strain Name'].str.split('(').str[0]
training_metadata[training_metadata['Strain Name'].isin(set(vaccine_strains.values()))]



