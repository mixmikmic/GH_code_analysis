'''
This notebook explore clusters of social landmarks that nyers follows
'''
import os
import pandas as pd
import numpy as np
import json
import pylab as pl
get_ipython().magic('pylab inline')

sl_strength = np.load(os.getenv('CUSSAC_OUTPUT') + '/social_landmark_strength.npy')

nonny_sl_strength = np.load(os.getenv('CUSSAC_OUTPUT') + '/nonny_social_landmark_strength.npy')

#Picks landmark at random and finds all other landmarks that are followed by it's followers
#
np.random.seed(179)
for i in np.random.choice(sl_strength.shape[1], 20):
    mutual_landmarks = np.array([])
    sl_followers = np.where(sl_strength[:,i] == 1)[0]
    for follower in sl_followers:
        mutual_landmarks = np.append(mutual_landmarks, np.where(sl_strength[follower,:] == 1)[0])
        
     
    print 'Social Landmark: ' + str(i),'Total Landmarks: ' + str(len(mutual_landmarks)), 'Common Landmarks: '         + str( (np.bincount(map(int,mutual_landmarks))>1).sum())
        

np.random.seed(179)
for i in np.random.choice(nonny_sl_strength.shape[1], 20):
    mutual_landmarks = np.array([])
    sl_followers = np.where(nonny_sl_strength[:,i] == 1)[0]
    for follower in sl_followers:
        mutual_landmarks = np.append(mutual_landmarks, np.where(nonny_sl_strength[follower,:] == 1)[0])
    print 'Social Landmark: ' + str(i),'Total Landmarks: ' + str(len(mutual_landmarks)), 'Common Landmarks: '         + str( (np.bincount(map(int,mutual_landmarks))>1).sum())
        


#pl.imshow(sl_strength, cmap = 'Greys')

for key in ny_friends.iterkeys():
    indices = (social_landmarks[social_landmarks['id'].isin(ny_friends[key][0])]).index.values
    sl_strength[nyer_lookup[nyer_lookup['screen_name'] == key ].index.values[0], indices]+=1
    

