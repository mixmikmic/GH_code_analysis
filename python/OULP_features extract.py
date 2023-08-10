import os

import numpy as np
import tensorflow as tf

from tensorflow_vgg import vgg16
from tensorflow_vgg import utils
from PIL import Image 
import cv2
import pickle

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
from scipy.ndimage import imread

data_dir_1 = r'OULP-C1V2_Pack/OULP-C1V2_NormalizedSilhouette(224x224x3)'
Seq = os.listdir(data_dir_1)
print(Seq)
sequences = [each for each in Seq if os.path.isdir(data_dir_1 + '/' + each)]
print(sequences)

from numpy import genfromtxt
IDList_A_55_probe = genfromtxt('OULP-C1V2_Pack/OULP-C1V2_SubjectIDList(FormatVersion1.0)/IDList_OULP-C1V2-A-75_gallery.csv',
           delimiter=',', dtype=np.int32)

# IDList_A_55_probe = genfromtxt('OULP-C1V2_Pack/OULP-C1V2_SubjectIDList(FormatVersion1.0)/IDList_OULP-C1V2-A-75_probe.csv',
#            delimiter=',', dtype=np.int32)

print('length:',len(IDList_A_55_probe))
for row in IDList_A_55_probe:
    print(row)

x = 0 
for i,j in zip(IDList_A_55_gallery, IDList_A_55_probe):
    if i[0] != j[0]:
        x += 1
print(x)

def generate_gait_cycle_list(folder_path, csv_file):
    for subject in csv_file:
        images_list = np.ndarray((subject[3] - subject[2] + 1, 224, 224, 3))
    
        subject_dir = str(subject[0])
        if len(subject_dir) != 7:
            subject_dir = '0' * (7 - len(subject_dir)) + subject_dir
                
        for ii, image_name in enumerate(range(subject[2], subject[3]+1)):        
            image_name = str(image_name)
            if len(image_name) == 2:
                image_name = '0' + image_name
            elif len(image_name) == 1:
                image_name = '00' + image_name
            images_list[ii, :, :, :] = imread(os.path.join(folder_path, subject_dir, '00000{}.png'.format(image_name)))
            
            if ii == 0:
                begin_image = image_name
                
        yield (images_list, subject_dir, begin_image, image_name)

Or_folder_path = 'OULP-C1V2_Pack/OULP-C1V2_NormalizedSilhouette(224x224x3)/Seq00'
# subject_list = os.listdir(Or_folder_path) 
# subject_list = subject_list[1:]
# save_folder = 'OULP-C1V2_Pack/OULP-deference-(128x88)/Seq00'
image_list_tuple = generate_gait_cycle_list(Or_folder_path, IDList_A_55_probe)

total = []
for ii in range(len(IDList_A_55_probe)):
    next_input = next(image_list_tuple)
    total.append(next_input[0])

total = np.array(total)

with open(r"gait_data/gallery_75degree_1_375", 'wb') as outfile:
    np.savez(outfile, total)

codes_list = []

codes = None

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope("content_vgg"):
        vgg.build(input_)
    
    for each in sequences:
        print("Starting {} sequence".format(each))
        sequence_path = data_dir_1 + '/' + each
        subjects = os.listdir(sequence_path)
        subjects = np.array(subjects)
        subjects = subjects[1:]
        for subject in subjects:
            subject_path = sequence_path + '/' + subject
            if subject == '0000000':
                files = os.listdir(subject_path)
                files.remove('.DS_Store')
#             elif subject == '0000024':
#                 files = os.listdir(subject_path)
#                 files.remove('.DS_Store')
            else:
                files = os.listdir(subject_path)
            
            for ii, file in enumerate(files, 1):
                img = imread(os.path.join(subject_path, file))
                batch.append(img.reshape((1, 224, 224, 3)))
            
                if ii % batch_size == 0 or ii == len(files):
                    images = np.concatenate(batch)
                
                    feed_dict = {input_: images}
                    codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)
                
                    if codes is None:
                        codes = codes_batch
                    else:
                        coeds = np.concatenate((codes, codes_batch))
                    
                    # Reset to start building the next batch
                    batch = []
                    print('{} images processed'.format(ii))



