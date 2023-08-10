import os
import diff_classifier.imagej as ij
import boto3
import os.path as op
import diff_classifier.aws as aws
import cloudknot as ck
import diff_classifier.knotlets as kn
import numpy as np

folder = '01_18_Experiment'

missing = []
for i in range(10, 15):
    missing.append("P1_S2_R_00{}".format(i))

for i in range(10, 15):
    missing.append("P2_S3_L_00{}".format(i))
    
for i in range(0, 15):
    missing.append("P3_S3_L_{}".format("%04d" % i))

pups = ["P1", "P2", "P3"]
slices = ["S1", "S2", "S3"]
folder = '01_18_Experiment'

hemis = ["R", "L"]

prefix = []
for pup in pups:
    for slic in slices:
        for hemi in hemis:
            for vid in range(0, 15):
                new = "{}_{}_{}_{}".format(pup, slic, hemi, "%04d" % vid)
                if not new in missing:
                    prefix.append(new)

s3 = boto3.client('s3')
for pref in prefix:
    fild = '{}/{}/{}.tif'.format(folder, pref.split('_')[0], pref)
    #print(fild)
    try:
        obj = s3.head_object(Bucket='ccurtis7.pup', Key=fild)
    except:
        print('Original image not found: {}'.format(fild))

s3 = boto3.client('s3')
for pref in prefix:
    fild = []
    split_counter = 0
    for row in range(0, 4):
        for col in range(0, 4):
            new = '{}/{}/{}_{}_{}.tif'.format(folder, pref.split('_')[0], pref, row, col)
            fild.append(new)
            try:
                obj = s3.head_object(Bucket='ccurtis7.pup', Key=new)
                split_counter = split_counter + 1
            except:
                counter = 1
    print('Successful partitioned videos for {}: {}'.format(pref, split_counter))

s3 = boto3.client('s3')
for pref in prefix:
    fild = []
    split_counter = 0
    for row in range(0, 4):
        for col in range(0, 4):
            new = '{}/{}/Traj_{}_{}_{}.csv'.format(folder, pref.split('_')[0], pref, row, col)
            fild.append(new)
            try:
                obj = s3.head_object(Bucket='ccurtis7.pup', Key=new)
                split_counter = split_counter + 1
            except:
                counter = 1
    print('Successful tracked videos for {}: {}'.format(pref, split_counter))

pref = prefix[0]
new = '{}/{}/{}_{}_{}.tif'.format(folder, pref.split('_')[0], pref, row, col)
name = op.split(new)[1]
DIR = op.abspath('.')
aws.download_s3(new, op.join(DIR, name))

import skimage.io as sio

test = sio.imread(op.join(DIR, name))

np.mean(test[0, :, :])

s3 = boto3.client('s3')
not_here = 0
for pref in prefix:
    fild = '{}/{}/msd_{}.csv'.format(folder, pref.split('_')[0], pref)
    #print(fild)
    try:
        obj = s3.head_object(Bucket='ccurtis7.pup', Key=fild)
        print('Successful MSD calculations for {}'.format(fild))
    except:
        print('MISSING MSD file: {}'.format(fild))
        not_here = not_here + 1
print('Total missing files: {}'.format(not_here))

s3 = boto3.client('s3')
not_here = 0
for pref in prefix:
    fild = '{}/{}/features_{}.csv'.format(folder, pref.split('_')[0], pref)
    #print(fild)
    try:
        obj = s3.head_object(Bucket='ccurtis7.pup', Key=fild)
        print('Successful feature calculations for {}'.format(fild))
    except:
        print('MISSING feature file: {}'.format(fild))
        not_here = not_here + 1
print('Total missing files: {}'.format(not_here))



