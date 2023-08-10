import numpy as np
import sys
import csv
import scipy.io as sio
from scipy.fftpack import fft, ifft
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
import operator
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

get_ipython().magic('matplotlib inline')

#base_dir = '/home/ibanez/data/amnh/darwin_notes/'
base_dir = '/data/amnh/darwin/'
curves_fft_dir = base_dir + 'image_csvs_fft/'
fft_similarity_dir = base_dir + 'fft_similarity_clean/'
base_image_dir = base_dir + 'images/'
base_fft_dir = base_dir + 'image_csvs_fft/'
base_csv_dir = base_dir + 'image_csvs/'

top_matches = pd.read_csv(base_dir + 'top_items_sorted.txt', index_col=False, header=None, sep=' ');
top_matches.columns = ["image1","image2","fft_score"]
top_matches.head()

def save_match(row_index):
    with open("/data/amnh/darwin/confirmed_matches.csv", "a+") as f:        
        image1_basename = top_matches["image1"][row_index]
        image2_basename = top_matches["image2"][row_index]
        fft_score = top_matches["fft_score"][row_index]
        print(image1_basename, image2_basename, fft_score)
        image1_filename = image1_basename[:-6] + '.jpg'
        image2_filename = image2_basename[:-14] + '.jpg'
        print(image1_filename)
        print(image2_filename)
        
        if 'south' in image1_basename:
            f.write("{},{},{}\n".format(image2_filename, image1_filename, fft_score))
        else:
            f.write("{},{},{}\n".format(image1_filename, image2_filename, fft_score))

def check_match_curves(row_index):
    image1_basename = top_matches["image1"][row_index]
    image2_basename = top_matches["image2"][row_index]
    fft_score = top_matches["fft_score"][row_index]
    fft1_filename = base_fft_dir + image1_basename + '_fft.mat'
    fft2_filename = base_fft_dir + image2_basename
    curve1_filename = base_csv_dir + image1_basename + '.csv'
    curve2_filename = base_csv_dir + image2_basename[:-8] + '.csv'
    if 'south' in image1_basename and 'south' in image2_basename:
        print('CONFLICTING BORDERS!')
        return
    if 'north' in image1_basename and 'north' in image2_basename:
        print('CONFLICTING BORDERS!')
        return
    fft1 = sio.loadmat(fft1_filename)['fft']
    fft2 = sio.loadmat(fft2_filename)['fft']
    curve1restored = np.real(ifft(fft1))
    curve2restored = np.real(ifft(fft2))
    curve1xy = pd.read_csv(curve1_filename)
    curve2xy = pd.read_csv(curve2_filename)
    curve1xyn = curve1xy - curve1xy.mean()
    curve2xyn = curve2xy - curve2xy.mean()
    curve1y = curve1xyn.ix[:,1] 
    curve2y = curve2xyn.ix[:,1]
    commonsize = min(curve1y.size, curve2y.size)
    curve1yt = curve1y[:commonsize]
    curve2yt = curve2y[:commonsize]
    rms = sqrt(mean_squared_error(curve1yt,curve2yt))
    print(rms)
    print(curve1_filename)
    print(curve2_filename)
    plt.figure()
    plt.plot(curve1y)
    plt.plot(curve2y)

def compute_match_curves(row_index):
    image1_basename = top_matches["image1"][row_index]
    image2_basename = top_matches["image2"][row_index]
    fft_score = top_matches["fft_score"][row_index]
    image1_filename = image1_basename[:-6] + '.jpg'
    image2_filename = image2_basename[:-14] + '.jpg'
    fft1_filename = base_fft_dir + image1_basename + '_fft.mat'
    fft2_filename = base_fft_dir + image2_basename
    curve1_filename = base_csv_dir + image1_basename + '.csv'
    curve2_filename = base_csv_dir + image2_basename[:-8] + '.csv'
    curve1xy = pd.read_csv(curve1_filename)
    curve2xy = pd.read_csv(curve2_filename)
    curve1xyn = curve1xy - curve1xy.mean()
    curve2xyn = curve2xy - curve2xy.mean()
    curve1y = curve1xyn.ix[:,1] 
    curve2y = curve2xyn.ix[:,1]
    commonsize = min(curve1y.size, curve2y.size)
    curve1yt = curve1y[:commonsize]
    curve2yt = curve2y[:commonsize]
    pow1 = sqrt((curve1yt**2).sum())
    pow2 = sqrt((curve2yt**2).sum())
    conflict = False
    verified = False
    rms = sqrt(mean_squared_error(curve1yt,curve2yt))
    if 'south' in image1_basename and 'south' in image2_basename:
        conflict = True
    if 'north' in image1_basename and 'north' in image2_basename:
        conflict = True
    return rms, fft_score, min(pow1, pow2), row_index, image1_filename, image2_filename, conflict, verified

rmss = [compute_match_curves(x) for x in range(0,2000)]

review = pd.DataFrame(sorted(rmss,key=operator.itemgetter(3),reverse=True),
                     columns=["rms", "fft_score", "minpow1pow2", "row_index", "image1_filename", "image2_filename", "conflict", "verified"])

review = review[~review['conflict']]

review.head()

plt.scatter(review['rms'], review['fft_score'])

matches = pd.read_csv(base_dir + 'confirmed_matches.csv',
                     names = ['image1_filename', 'image2_filename', 'fft_score'])

matches.head()

image_matches_index = pd.DataFrame(matches['image1_filename']+':'+matches['image2_filename'], columns=['image_index'])

image_review_index = pd.DataFrame(review['image2_filename']+':'+review['image1_filename'], columns=['image_index'])

for index, row in review.iterrows():
    verified = False
    image_index = row['image1_filename'] + ':' + row['image2_filename']
    if image_index in image_matches_index.image_index.values:
        verified = True
    image_index = row['image2_filename'] + ':' + row['image1_filename']
    if image_index in image_matches_index.image_index.values:
        verified = True
    review.set_value(col='verified', index=index, value=verified)

review.to_csv(base_dir + 'parametric_matches.csv', header=True, index=False,
              columns=["rms", "fft_score", "minpow1pow2", "row_index", "image1_filename", "image2_filename", "verified"])

reviewverified = review[review['verified']]
min_fft_score = min(reviewverified['fft_score'])
max_rms = max(reviewverified['rms'])
min_minpower = min(reviewverified['minpow1pow2'])

candidates = review[review['verified']==False]
candidates = candidates[candidates['rms'] < max_rms]
candidates = candidates[candidates['fft_score'] > min_fft_score]
candidates = candidates[candidates['minpow1pow2'] > min_minpower]

candidates.sort_values('row_index', ascending=False)

check_match_curves(368)

save_match(368)



