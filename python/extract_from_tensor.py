import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

# - iterate over tensorflow record, one example at a time
num_bad_labels = 0
num_videos_in_record = 0

import glob
filenames = glob.glob('data/*.tfrecord')
featureDict = {}
for video_level_data in filenames:
    print('processing ',video_level_data)
    
    num_videos_in_part = 0
    for example in tf.python_io.tf_record_iterator(video_level_data):
        num_videos_in_record += 1
        num_videos_in_part += 1
        # yield example
        tf_example = tf.train.Example.FromString(example)

        # get video id from example
        vid_id = tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8')
        
        # get list of labels from example
        label_idx_list = tf_example.features.feature["labels"].int64_list.value

        # instantiate list for KEY=vid_id
        featureDict[vid_id] = label_idx_list
   

    print('############# {} VIDEOS IN RECORD ##############'.format(num_videos_in_part))                    
print('\nAnalyzed {} videos from YT8M'.format(num_videos_in_record))

categories =  {122 : "Weight training",
               204 : "Gym",
               529 : "Squat",
               532 : "Barbell",
              898 : "Pilates",
               1389 : "Dumbell",
               1430 : "Pull Up",           
               2044 : "Bodyweight",
               2045 : "Bicep Curls",
               4441 : "Functional Training",
              4667 : "Golds Gym"}

# label 204 is gym
gymVids = set()
for val in featureDict:
    for cat in categories:
        if cat in featureDict[val]:
            print(val)
            gymVids.add(val)

print(len(gymVids))

import csv
import urllib.request as urllib2
from lxml import html, etree
import json

api_key = 'AIzaSyCrFWiPfGcb5IsyS-wpAMk6eaNdMaC8pXs'
channel = 'UCF0pVplsI8R5kcAqgtoRqoA'
vidStats =  'https://www.googleapis.com/youtube/v3/videos?part=id,statistics&id='
vidSnips = 'https://www.googleapis.com/youtube/v3/videos?part=id,snippet&id='

firstTime = False
with open('videoStats.csv', 'a',newline='') as c:

    writer = csv.writer(c)
    
    if firstTime:
        writer.writerow(['Id', 
                     'Title', 
                     'Description', 
                     'LikeCount', 
                     'DislikeCount', 
                     'ViewCount', 
                     'FavoriteCount', 
                     'CommentCount', 
                     'PublishedAt', 
                     'Channel Id', 
                     'Channel Title',
                     'Tags',
                    'Thumbnail Default'])

    gymVids = list(gymVids)
    counter = 0;
    for vid in gymVids[2275:]:
        stats = json.load(urllib2.urlopen(vidStats + vid + '&key=' + api_key))
        snips = json.load(urllib2.urlopen(vidSnips + vid + '&key=' + api_key))
        if(len(stats['items'])==0):
            print("Could not find: " +  str(vid))
            continue
        s = stats['items'][0]['statistics']
        sn = snips['items'][0]
        try:
            writer.writerow([sn['id'], 
                      sn['snippet']['title'].encode('utf8'), 
                     sn['snippet']['description'].encode('utf8'), 
                     s['likeCount'],
                     s['dislikeCount'], 
                     s['viewCount'], 
                     s['favoriteCount'], 
                     s['commentCount'], 
                     sn['snippet']['publishedAt'],
                     sn['snippet']['channelId'],
                     sn['snippet']['channelTitle'],
                     sn['snippet']['tags'],
                     sn['snippet']['thumbnails']['default']['url'].encode("utf8")])
        except:
            print("Skipped video: " + str(vid))
        counter +=1
        if(counter % 5 ==0):
            print("Progess: " + str(counter))
print("Download Completed")



