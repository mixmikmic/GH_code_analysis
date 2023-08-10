import numpy as np
import sys
import scipy.io as sio
import os
import operator
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image

base_dir = '/data/amnh/darwin/'
base_image_dir = base_dir + 'images/'

fft_norms = pd.read_csv(base_dir + 'fft_norms.csv', index_col=False, header=None, sep='\t');
fft_norms.columns = ["image1", "fft"]
fft_norms.drop(fft_norms.index[:10], inplace=True)
fft_norms.head(10)

fft_norms.plot()
plt.show()

fft_top_matches = pd.read_csv(base_dir + 'top_items_sorted.txt', index_col=False, header=None, sep=' ');
fft_top_matches.columns = ["image1", "image2", "fft_score"]
fft_top_matches.head()

fft_top_interesting_matches = pd.merge(fft_norms, fft_top_matches, how='inner')
fft_top_interesting_matches.columns = ["image1","fft1","best_match","fft_score"]
fft_top_interesting_matches.sort_values(by=["fft1","fft_score"])
fft_top_interesting_matches.head()

# Sanity check: 
# we find the index of a known match
print(fft_top_interesting_matches.loc[fft_top_interesting_matches["image1"]=="MS-DAR-00084-00002-000-00307_north_fft.mat"])

row_index = 658
image1_basename = fft_top_interesting_matches["image1"][row_index]
image2_basename = fft_top_interesting_matches["best_match"][row_index]
fft1_value = fft_top_interesting_matches["fft1"][row_index]
fft_score = fft_top_interesting_matches["fft_score"][row_index]
print(image1_basename, image2_basename, fft1_value, fft_score)
image1_filename = base_image_dir + image1_basename[:-14] + '.jpg'
image2_filename = base_image_dir + image2_basename[:-14] + '.jpg'

Image(filename=image1_filename, height=0.5)

Image(filename=image2_filename)

fft_top_interesting_matches.plot(kind='scatter', x='fft1', y='fft_score')
plt.show()



