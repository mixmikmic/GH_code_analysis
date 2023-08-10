import numpy as np
import sys
import scipy.io as sio
import os
import operator
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

base_dir = '/data/amnh/darwin/'
curves_fft_dir = base_dir + 'image_csvs_fft/'
fft_similarity_dir = base_dir + 'fft_similarity_clean/'
base_image_dir = base_dir + 'images/'

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

def check_match(row_index):
    image1_basename = top_matches["image1"][row_index]
    image2_basename = top_matches["image2"][row_index]
    fft_score = top_matches["fft_score"][row_index]
    print(image1_basename, image2_basename, fft_score)
    image1_filename = base_image_dir + image1_basename[:-6] + '.jpg'
    image2_filename = base_image_dir + image2_basename[:-14] + '.jpg'
    print(image1_filename)
    print(image2_filename)
    image1 = Image(filename=image1_filename, width=700)
    image2 = Image(filename=image2_filename, width=700)
    if 'south' in image1_basename:
        if 'south' in image2_basename:
            print('CONFLICTING BORDERS!')
        else:
            display(image2, image1)
    else:
        if 'north' in image2_basename:
            print('CONFLICTING BORDERS!')
        else:
            display(image1, image2)

check_match(4)

save_match(4)



