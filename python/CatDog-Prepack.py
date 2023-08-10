from os import getcwd, rename
from shutil import copyfile
from numpy.random import choice
from glob import glob
from scipy.misc import imread, imresize, imsave
from tqdm import tqdm
from vaiutils import path_consts

for k,v in path_consts('CatDog-Prepack', 'CatDog'):
    exec(k+'=v')

FRACTION_VALID = 0.2
FRACTION_SAMPLE = 0.01

DIR_TRAIN = DIR_DATA + '/train'
DIR_TEST = DIR_DATA + '/test'
DIR_VALID = DIR_DATA + '/valid'
DIR_RESULTS = DIR_DATA + '/results'

DIR_SAMPLE_TRAIN = DIR_DATA + '/sample/train'
DIR_SAMPLE_TEST = DIR_DATA + '/sample/test'
DIR_SAMPLE_VALID = DIR_DATA + '/sample/valid'
DIR_SAMPLE_RESULTS = DIR_DATA + '/sample/results'

def resize_img(path):
    get_ipython().run_line_magic('cd', '$path')
    filenames = glob('*.jpg')
    for filename in tqdm(filenames):
        imsave(path + '/' + filename, imresize(imread(path + '/' + filename), (224, 224)))

resize_img(DIR_SAMPLE_VALID + '/cats')
resize_img(DIR_SAMPLE_VALID + '/dogs')
resize_img(DIR_SAMPLE_TRAIN + '/cats')
resize_img(DIR_SAMPLE_TRAIN + '/dogs')
resize_img(DIR_VALID + '/cats')
resize_img(DIR_VALID + '/dogs')
resize_img(DIR_TRAIN + '/cats')
resize_img(DIR_TRAIN + '/dogs')
resize_img(DIR_TEST + '/unknown')
resize_img(DIR_SAMPLE_TEST + '/unknown')

get_ipython().run_line_magic('cd', '$DIR_DATA')
for path in ['test/unknown', 'valid', 'results', 'sample/train', 'sample/test/unknown', 'sample/valid', 'sample/results']:
    get_ipython().run_line_magic('mkdir', '-p $path')

def move_img(from_path, to_path, fraction, copy=False):
    get_ipython().run_line_magic('cd', '$from_path')
    
    filenames = glob('*.jpg')
    filenames = choice(filenames, int(fraction*len(filenames)), replace=False)
    
    for filename in filenames:
        if copy:
            copyfile(from_path + '/' + filename, to_path + '/' + filename)
        else:
            rename(from_path + '/' + filename, to_path + '/' + filename)
        
move_img(DIR_TRAIN, DIR_VALID, FRACTION_VALID)
move_img(DIR_TRAIN, DIR_SAMPLE_TRAIN, FRACTION_SAMPLE, copy=True)
move_img(DIR_VALID, DIR_SAMPLE_VALID, FRACTION_SAMPLE, copy=True)
move_img(DIR_TEST, DIR_SAMPLE_TEST + '/unknown', FRACTION_SAMPLE, copy=True)

def separate_cats_dogs(path):
    get_ipython().run_line_magic('cd', '$path')
    get_ipython().run_line_magic('mkdir', 'cats')
    get_ipython().run_line_magic('mkdir', 'dogs')
    get_ipython().run_line_magic('mv', 'cat.*.jpg cats/')
    get_ipython().run_line_magic('mv', 'dog.*.jpg dogs/')
    
separate_cats_dogs(DIR_TRAIN)
separate_cats_dogs(DIR_VALID)
separate_cats_dogs(DIR_SAMPLE_TRAIN)
separate_cats_dogs(DIR_SAMPLE_VALID)

get_ipython().run_line_magic('cd', '$DIR_TEST')
get_ipython().run_line_magic('mv', '*.jpg unknown/')

def join_cats_dogs(path):
    get_ipython().run_line_magic('cd', '$path/cats')
    get_ipython().run_line_magic('mv', '*.jpg ../')
    get_ipython().run_line_magic('cd', '$path/dogs')
    get_ipython().run_line_magic('mv', 'dog.*.jpg ../')

join_cats_dogs(DIR_TRAIN)
join_cats_dogs(DIR_VALID)
join_cats_dogs(DIR_SAMPLE_TRAIN)
join_cats_dogs(DIR_SAMPLE_VALID)

get_ipython().run_line_magic('cd', '$DIR_TEST/unknown')
get_ipython().run_line_magic('mv', '*.jpg ../')
get_ipython().run_line_magic('cd', '$DIR_SAMPLE_TEST')
get_ipython().run_line_magic('mv', '*.jpg ../')

