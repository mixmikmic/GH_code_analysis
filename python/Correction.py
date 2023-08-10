get_ipython().magic('pylab inline')

import sys
sys.path.append('../')
sys.path.append('../support/')
from glob import glob
from os.path import join, isfile, basename
from multiprocessing import Pool
from scipy.ndimage.interpolation import rotate
from IPython.display import clear_output
from ct_reader import *
from tqdm import tqdm
from functools import partial
from matplotlib.pyplot import *
import pickle
from paths import *
from scipy.misc import imresize

BATCH_SIZE = 10

def read_ct(path, ret_xy_spacing=False):
    patient = read_ct_scan(path)
    image = get_pixels_hu(patient)
    image[image == image[0,0,0]] = 0
    
    if ret_xy_spacing:
        try:
            return image, patient[0].PixelSpacing[0]
        except AttributeError:
            return image, scan.GetSpacing()[0]
    
    return image

def display(patient, mask):
    
    mask[(mask == 4) 
         | (mask == 12) 
         | (mask == 8)] = 0
    
    mask[(mask == 1) 
         | (mask == 5) 
         | (mask == 9)
         | (mask == 13)] = 1
    
    mask[(mask == 2) 
         | (mask == 6) 
         | (mask == 10)
         | (mask == 14)] = 2
    
    mask[(mask == 3) 
         | (mask == 7) 
         | (mask == 15)] = 3
    
    subplot(2, 2, 1)
    imshow(patient[patient.shape[0] // 2])
    axis('off')
    subplot(2, 2, 2)
    imshow(imresize(clip(patient[:, patient.shape[1] // 2], -1000, 400),
                    (patient.shape[0], patient.shape[0])))
    axis('off')
    subplot(2, 2, 3)
    imshow(mask[patient.shape[0] // 2])
    axis('off')
    subplot(2, 2, 4)
    imshow(imresize(mask[:, patient.shape[1] // 2], 
                    (patient.shape[0], patient.shape[0])))
    axis('off')
    show()

global_paths = glob(join(PATH['STAGE_MASKS'], "*[0-9a-f].npy"))
global_paths = sorted([join(PATH['STAGE_DATA'], basename(path).split('.npy')[0]) for path in global_paths])
erroneus = list()
upsides = list()
checkpoint = 0
iterations = int(ceil(len(global_paths) / BATCH_SIZE))

erroneus = list()
iterations = int(ceil(len(global_paths) / BATCH_SIZE))
for counter in range(checkpoint, iterations):
    paths = global_paths[BATCH_SIZE * counter:
                         BATCH_SIZE * (counter + 1)]
    for i, path in enumerate(paths):
        patient = read_ct(path)
        mask = load(join(PATH['STAGE_MASKS'], 
                         basename(path) + '.npy'))
        
        print(i, iterations - counter, path)
        display(patient, mask)
        
    while True:
        try:
            print('Erroneus:')
            err = input()
            nomerus = list()
            if err != '':
                nomerus = list(map(int, err.split(' ')))
            print('Inverted:')
            ups = input()
            nomerus = [nomerus, []]
            if ups != '':
                nomerus[1] = list(map(int, ups.split(' ')))
            break
        except:
            pass

    for i in nomerus[0]:
        erroneus.append(basename(paths[abs(i)]))
    pickle.dump(erroneus, 
                open(join(PATH['STAGE_MASKS'], 
                          'still_erroneus_ncrash'), 'wb'))
    
    for i in nomerus[1]:
        upsides.append(basename(paths[abs(i)]))
    pickle.dump(upsides, 
                open(join(PATH['STAGE_MASKS'], 
                          'upsides'), 'wb'))
    clear_output()
    

erroneus = pickle.load(open(join(PATH['STAGE_MASKS'], 
                      'still_erroneus_ncrash'), 'rb'))
upsides = pickle.load(open(join(PATH['STAGE_MASKS'], 
                      'upsides'), 'rb'))

