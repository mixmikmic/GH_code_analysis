#!pip install --user --upgrade watson-developer-cloud

#Making a local folder to put my data.

#NOTE: YOU MUST do something like this on a Spark Enterprise cluster at the hackathon so that
#you can put your data into a separate local file space. Otherwise, you'll likely collide with 
#your fellow participants. 

my_team_name_data_folder = 'my_team_name_data_folder'

mydatafolder = os.environ['PWD'] + '/' +  my_team_name_data_folder + '/zipfiles'
if os.path.exists(mydatafolder) is False:
    os.makedirs(mydatafolder)

get_ipython().system('ls my_team_name_data_folder/zipfiles')

from __future__ import division

import cStringIO
import glob
import json
import os
import requests
import time
import timeit
import zipfile
import copy

from random import randint

import matplotlib.pyplot as plt
import numpy as np

import ibmseti

from watson_developer_cloud import VisualRecognitionV3

apiVer = VisualRecognitionV3.latest_version #'2016-05-20'
classifier_prefix = 'setisignals'

#You can sign up with WatsonVR through Bluemix to get a key
#However, Hackathon participants will be provided with a WATSON VR key that has more free API calls per day.
apiKey = 'WATSON-VISUAL-RECOGNITION-API-KEY'  

vr = VisualRecognitionV3(apiVer, api_key=apiKey)

## View all of your classifiers

classifiers = vr.list_classifiers()
print json.dumps(classifiers, indent=2)

## Run this cell ONLY IF you want to REMOVE all classifiers
# Otherwise, the subsequent cell will append images to the `classifier_prefix` classifier
classifiers = vr.list_classifiers()
for c in classifiers['classifiers']:
    vr.delete_classifier(c['classifier_id'])

classifiers = vr.list_classifiers()
print json.dumps(classifiers, indent=2)

#Create new classifier, or get the ID for the latest SETISIGNALS classifier

classifier_id = None
classifier = None

classifiers = vr.list_classifiers()

for c in classifiers['classifiers']:
    if c['status'] == 'ready' and (classifier_prefix in c['classifier_id']):
        classifier_id = c['classifier_id']


if classifier_id is not None:
    classifier = vr.get_classifier(classifier_id)
    print '\r\nFound classifer:\r\n\r\n{}'.format(json.dumps(classifier, indent=2))
else:
    print 'No custom classifier available\r\n'
    print(json.dumps(classifiers, indent=2))

squiggle = sorted(glob.glob('{}/classification_*_squiggle.zip'.format(mydatafolder)))
narrowband = sorted(glob.glob('{}/classification_*_narrowband.zip'.format(mydatafolder)))
narrowbanddrd = sorted(glob.glob('{}/classification_*_narrowbanddrd.zip'.format(mydatafolder)))
noise = sorted(glob.glob('{}/classification_*_noise.zip'.format(mydatafolder)))

sq = len(squiggle)
nb = len(narrowband)
nd = len(narrowbanddrd)
ns = len(noise)

## Possible todo here: Try using the 'noise' as a "negative" example when training Watson. See the Watson documentation.

num = max(sq, nb, nd, ns)
#num = max(sq, nb, nd)

if classifier_id is None:
    print 'Adding custom classifier ... this may take awhile'
else:
    print 'Updating custom classifier {} ... this may take awhile'.format(classifier_id)

for i in range(num):
    squiggle_p = open(squiggle[i], 'rb') if i < sq else None
    narrowband_p = open(narrowband[i], 'rb') if i < nb else None
    narrowbanddrd_p = open(narrowbanddrd[i], 'rb') if i < nd else None
    noise_p = open(noise[i], 'rb') if i < ns else None

    if classifier_id is None:
#        print 'Creating with\r\n{}\r\n{}\r\n{}\r'.format(squiggle_p, narrowband_p, narrowbanddrd_p)  #use this line if going to use 'noise' as negative example
        print 'Creating with\r\n{}\r\n{}\r\n{}\r\n{}\r'.format(squiggle_p, narrowband_p, narrowbanddrd_p, noise_p)
        classifier = vr.create_classifier(
            classifier_prefix,
            squiggle_positive_examples = squiggle_p,
            narrowband_positive_examples = narrowband_p,
            narrowbanddrd_positive_examples = narrowbanddrd_p,
            noise_positive_examples = noise_p  #remove this if going to use noise as 'negative' examples
        )
        
        classifier_id = classifier['classifier_id']
    else:
        print 'Updating with\r\n{}\r\n{}\r\n{}\r\n{}\r'.format(squiggle_p, narrowband_p, narrowbanddrd_p, noise_p)
#        print 'Updating with\r\n{}\r\n{}\r\n{}\r'.format(squiggle_p, narrowband_p, narrowbanddrd_p)  #use this line if going to use 'noise' as negative example
        classifier = vr.update_classifier(
            classifier_id,
            squiggle_positive_examples = squiggle_p,
            narrowband_positive_examples = narrowband_p,
            narrowbanddrd_positive_examples = narrowbanddrd_p,
            noise_positive_examples = noise_p #remove this if going to use noise as 'negative' examples
        )

    if squiggle_p is not None:
        squiggle_p.close()
    if narrowband_p is not None:
        narrowband_p.close()
    if narrowbanddrd_p is not None:
        narrowbanddrd_p.close()
    if noise_p is not None:
        noise_p.close()

    if classifier is not None:
        print('Classifier: {}'.format(classifier_id))
        status = classifier['status']
        startTimer = timeit.default_timer()
        while status in ['training', 'retraining']:
            print('Status: {}'.format(status))
            time.sleep(10)
            classifier = vr.get_classifier(classifier_id)
            status = classifier['status']
        stopTimer = timeit.default_timer()
        print '{} took {} minutes'.format('Training' if i == 0 else 'Retraining', int(stopTimer - startTimer) / 60)

print(json.dumps(vr.get_classifier(classifier_id), indent=2))

zz = zipfile.ZipFile(mydatafolder + '/' + 'testset_narrowband.zip')

test_list = zz.namelist()
randomSignal = zz.open(test_list[10],'r')

from IPython.display import Image
squigImg = randomSignal.read()
Image(squigImg)

#note - have to 'open' this again because it was already .read() out in the line above
randomSignal = zz.open(test_list[10],'r')

url_result = vr.classify(images_file=randomSignal, classifier_ids=classifier_id, threshold=0.0)

print(json.dumps(url_result, indent=2))

#Create a dictionary object to store results from Watson

from collections import defaultdict

class_list = ['squiggle', 'noise', 'narrowband', 'narrowbanddrd']

results_group_by_class = {}
for classification in class_list:
    results_group_by_class[classification] = defaultdict(list)
    
failed_to_classify_uuid_list = []

print classifier_id

results_group_by_class

### NOTE. If this breaks due to a requests timeout or other error: **just restart this cell**
#   The processing should pick up where it left off. 

##  NOTE: This code could be more efficient and make fewer HTTP calls to Watson. I could have dumped the testset_<class>.zip into 
#     smaller zip files (testset_<class>_N.zip for N = 1,2,3,4...) and then made a single call to Watson with each smaller zip file
#
#    Example:
#     with open(mydatafolder + '/' + 'testset_squiggle_1.zip', 'rb') as squigglezips:
#       url_result = vr.classify(images_file=squigglezips, classifier_ids=classifier_id, threshold=0.0)

#     The 'testset_squiggle.zip' files are too large to make a single to call to Watson, and so this code goes through
#     each file one by one.

### ASLO, I could have farmed this out to the Spark executor nodes as well. 

for sigclass in class_list:
    
    passed = 0
    
    zz = zipfile.ZipFile(mydatafolder + '/' + 'testset_{}.zip'.format(sigclass))
    zzlist = zz.namelist()
    
    ### REDUCING TESTING to only the first 30 signals in the test set -- to keep this demonstration code faster.
    
    zzlist = zzlist[:30]  
    
    zzlistsize = len(zzlist)
    
    startTimer = timeit.default_timer()

    resdict = results_group_by_class[classification]
    
    print 'Running test ({} images) for {}... this may take a while.'.format(zzlistsize, sigclass)

    for fn in zzlist:
        pngfilename = fn.split('/')[-1]
        uuid = pngfilename.split('.')[0]
        classification = sigclass
        
        if uuid in resdict['uuid'] or uuid in failed_to_classify_uuid_list:
            print "   have already classified {}".format(uuid)
            continue
        
        classify_result = vr.classify(images_file=zz.open(fn,'r'), classifier_ids=classifier_id, threshold=0.0)
        
        maxscore = 0
        maxscoreclass = None

        classifiers_arr = classify_result['images'][0]['classifiers']
        
        score_list = []
        for classifier_result in classifiers_arr:
            for class_result in classifier_result['classes']:
                score_list.append((class_result['class'],class_result['score']))
                if class_result['score'] > maxscore:
                    maxscore = class_result['score']
                    maxscoreclass = class_result['class']

        #sort alphabetically
        score_list.sort(key = lambda x: x[0])
        score_list = map(lambda x:x[1], score_list)

        if maxscoreclass is None:
            print 'Failed: {} - Actual: {}, No classification returned'.format(pngfilename, classification)
            #print(json.dumps(classify_result, indent=2))

        elif maxscoreclass != classification:
            print 'Failed: {} - Actual: {}, Watson Predicted: {} ({})'.format(pngfilename, classification, maxscoreclass, maxscore)
        else:
            passed += 1
            print 'Passed: {} - Actual: {}, Watson Predicted: {} ({})'.format(pngfilename, classification, maxscoreclass, maxscore)

        if maxscoreclass is not None:
            resdict['signal_classification'].append(classification)
            resdict['uuid'].append(uuid)
            resdict['watson_class'].append(maxscoreclass)
            resdict['watson_class_score'].append(maxscore)
            resdict['scores'].append(score_list)
        else:
            #add to failed list
            failed_to_classify_uuid_list.append(uuid)

    stopTimer = timeit.default_timer()

    print 'Test Score: {}% ({} of {} Passed)'.format(int((float(passed) / zzlistsize) * 100), passed, zzlistsize)
    print 'Tested {} images in {} minutes'.format(zzlistsize, int(stopTimer - startTimer) / 60)

print "DONE"
        

import pickle
pickle.dump(results_group_by_class, open(mydatafolder + '/' + "watson_results.pickle", "w"))

watson_results = pickle.load(open(mydatafolder + '/' + "watson_results.pickle","r"))
# reorganize the watson_results dictionary to extract
# a list of [true_class, [scores], estimated_class] and
# use these for measuring our model's performance

class_scores = []
for k in watson_results.keys():
    class_scores += zip(watson_results[k]['uuid'], watson_results[k]['signal_classification'], watson_results[k]['scores'], watson_results[k]['watson_class'] )

class_scores[100]

from sklearn.metrics import classification_report
import sklearn

y_train = [x[1] for x in class_scores]
y_pred = [x[3] for x in class_scores]
y_prob = [x[2] for x in class_scores]
#we normalize the Watson score values to 1 in order to use them in the log_loss calculation even though the Watson VR scores are not true class prediction probabilities
y_prob = map(lambda x: (x, sum(x)), y_prob)
y_prob = map(lambda x: [y / x[1] for y in x[0]], y_prob)

print sklearn.metrics.classification_report(y_train,y_pred)
print sklearn.metrics.confusion_matrix(y_train,y_pred)
print("Classification accuracy: %0.6f" % sklearn.metrics.accuracy_score(y_train,y_pred) )
print("Log Loss: %0.6f" % sklearn.metrics.log_loss(y_train,y_prob) )

import csv
my_output_results = my_team_name_data_folder + '/' + 'watson_scores.csv'
with open(my_output_results, 'w') as csvfile:
    fwriter = csv.writer(csvfile, delimiter=',')
    for row in class_scores:
        fwriter.writerow([row[0]] + row[2])

get_ipython().system('cat my_team_name_data_folder/watson_scores.csv')



