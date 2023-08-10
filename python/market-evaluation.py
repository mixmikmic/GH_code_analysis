MARKET_ROOT = '/media/hpc3_storage/eustinova/dataset/Market-1501-v15.09.15'
caffe_root='/home/eustinova/caffe-master/'

lmdb_train_path = '/media/hpc2_storage/eustinova/lmdb/market_1501_160_60_10_overlap/train'

### import random
from scipy.spatial.distance import cosine, cdist
import matplotlib.pyplot as plt
get_ipython().run_line_magic('pylab', 'inline')

import lmdb
import time
import numpy as np


import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe.io import caffe_pb2
import caffe.io 
import os
import cv2

import random

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
import struct

def cos_dist(x, y):
    xy = np.dot(x,y);
    xx = np.dot(x,x);
    yy = np.dot(y,y);  
            
    return -xy*1.0/np.sqrt(xx*yy);

def getDistances(gescr_query, gescrs_gallery):
    dist = list()
    for i in xrange(len(gescrs_gallery)):
        dist.append(cos_dist(gescr_query, gescrs_gallery[i]))
        
    return dist 

def ifJunk(filename):
    if filename.startswith("-1"):
        return True;
    else :
        return False;
    
def ifDistractor(filename):
    if filename.startswith("0000"):
        return True;
    else :
        return False;
# pos 1, neg 0, junk -1
def assignJunkPosNeg(query_filename, bounding_box_test_file_name, Index):    
    person, camera = parse_market_1501_name(query_filename)
    labeling = list()   
    
    for i in xrange(len(bounding_box_test_file_name)):
        if bounding_box_test_file_name[i] in Index['junk'] or bounding_box_test_file_name[i] in Index[query]['junk']:
            labeling.append(-1)
        elif     bounding_box_test_file_name[i] in Index['distractor']:
            labeling.append(0)
        elif bounding_box_test_file_name[i] in Index[query]['pos']:
            labeling.append(1)
        else :
            labeling.append(0)
            
            
def getPlace(query, sorted_gallery_filenames, Index):    
        
    place = 0
    for i in xrange(len(sorted_gallery_filenames)):
        if sorted_gallery_filenames[i] in Index['junk'] or sorted_gallery_filenames[i] in Index[query]['junk']:
            continue
        elif     sorted_gallery_filenames[i] in Index['distractor']:
            place +=1
           
        elif sorted_gallery_filenames[i] in Index[query]['pos']:
           # print "PLACE " , sorted_gallery_filenames[i]
            return place
        else :
            place +=1
            
    return place        
def getAveragePrecision(query, sorted_gallery_filenames, Index):    
        
    ap = 0
    tp = 0
    k = 0
    
    for i in xrange(len(sorted_gallery_filenames)):
        
        if sorted_gallery_filenames[i] in Index['junk'] or sorted_gallery_filenames[i] in Index[query]['junk']:
            continue
        elif     sorted_gallery_filenames[i] in Index['distractor']:
            k+=1
            deltaR = 0
        elif sorted_gallery_filenames[i] in Index[query]['pos']:
            tp+=1
            k+=1
            deltaR = 1.0/len(Index[query]['pos'])
        else :
            k +=1
            deltaR = 0
        
        if tp == len(Index[query]['pos']):
            return ap
        precision = tp*1.0/k * deltaR
        ap += precision
        
    return ap    
def mAP(gescrs_query, query_image_names, gescrs_gallery, test_image_names, maxrank, Index):
    ranks = np.zeros(maxrank+1)
    places = dict()
    
    ap_sum = 0
    for qind in xrange(len(gescrs_query)):       
        print qind
        dist = getDistances(gescrs_query[qind], gescrs_gallery)
        #gallery_permutation = sorted(range(len(dist)), key=lambda k: dist[k])
        dist_zip = sorted(zip(dist,test_image_names))
        gallery_names_sorted = [x for (y,x) in dist_zip]
      
        ap=getAveragePrecision(query_image_names[qind], gallery_names_sorted, Index)
        ap_sum += ap
        

        
        
    return ap_sum * 1.0 /len(gescrs_query)

def ranking(metric, gescrs_query, query_image_names, gescrs_gallery, test_image_names, maxrank, Index):
    ranks = np.zeros(maxrank+1)
    places = dict()
    print('Calculating distances')
    all_dist = cdist(gescrs_query, gescrs_gallery, metric)
    np_test_image_names = np.array(test_image_names)
    img_names_sorted = dict()
    
    all_gallery_names_sorted = np_test_image_names[np.argsort(all_dist).astype(np.uint32)]
    for qind in xrange(len(gescrs_query)):       
        #print qind
        #dist = getDistances(gescrs_query[qind], gescrs_gallery)
        dist = all_dist[qind]
        #gallery_permutation = sorted(range(len(dist)), key=lambda k: dist[k])
        #dist_zip = sorted(zip(dist,test_image_names))
        #gallery_names_sorted = [x for (y,x) in dist_zip]
        gallery_names_sorted = all_gallery_names_sorted[qind]
      
        place=getPlace(query_image_names[qind], gallery_names_sorted, Index)       
        img_names_sorted[qind] = all_gallery_names_sorted[qind]
                
        places[qind] = place

        ranks[place+1:maxrank+1] += 1
        
    return ranks, img_names_sorted,places

def getDescr(batch, net):
    data = np.array(batch) / 256.0 
    out = net.forward_all(data = data)
    res = [np.copy(out['ip1_reid'][i]) for i in xrange(len(batch))]
    return res

def getLargeDescr(data, net, batch_size = 128):
    result = []
    for i in xrange(len(data)/batch_size) :
        
        batch = data[i * batch_size : min(len(data), (i+1)*batch_size)]
        result.extend(getDescr(batch, net))
    
    batch = data[(i+1) * batch_size : min(len(data), (i+2)*batch_size)]
    result.extend(getDescr(batch, net))
    
    return result


def parse_market_1501_name(full_name):
    
    name_ar = full_name.split('/')
    name = name_ar[len(name_ar)-1]
    
    person = int(name.split('_')[0])
    camera = int(name.split('_')[1].split('s')[0].split('c')[1])
    
    
    return person, camera

def parseMarket1501(path):
    person_label = list()
    camera_label = list()
    image_path = list()
    image_name = list()

    for file in sorted(os.listdir(path)):
        if file.endswith(".jpg"):
            person, camera = parse_market_1501_name(file)
            
            person_label.append(person)
            camera_label.append(camera)
            image_path.append(os.path.join(path, file))
            image_name.append(file)
    return person_label, camera_label, image_path, image_name

def transpose_for_storage(im):
    return im.transpose((2, 0, 1))
def transpose_for_show(im):
    return im.transpose((1, 2, 0))                                                                                                        
def break_the_image(stripes_num, img, overlap = 0):
    parts = list();
    overlap_stripes_num = stripes_num - 1;
    
    part_height = (np.shape(img)[0] + overlap_stripes_num * overlap)/stripes_num;
    
    for i in range(stripes_num):
        
        start = i * part_height - max((i) * overlap, 0);
        end = min(np.shape(img)[0], start + part_height);
        current_part = np.copy(img[start:end][:][:])    
        parts.extend(transpose_for_storage(current_part));
        
    return np.array(parts);    
def prepareImage(im, transform_params):
    
    result = list()
    #for masks
    if transform_params['color_transformer']  != None :
        im = imgToMask(im, transform_params['color_transformer'])
        
    #for images and masks
    if transform_params['reshape_params'] != None :

        if transform_params['reshape_params']['resize'] != None :
            #for masks
            if transform_params['reshape_params']['interpolation'] != None :
                #print transform_params['reshape_params']['interpolation'], type(transform_params['reshape_params']['interpolation'])
                im = cv2.resize(im, transform_params['reshape_params']['resize'], interpolation = transform_params['reshape_params']['interpolation']) 

            else :
                im = cv2.resize(im, transform_params['reshape_params']['resize'])
        
        #for masks add one more dimension for compatibility 
        if im.ndim == 2 :
            im = im.reshape((im.shape[0], im.shape[1], 1))
            
        #ugly : this transposes so we can not avoid breaking the image    
        ## breake and transpose (chan, h, w)    
        if transform_params['reshape_params']['stripes'] != None and transform_params['reshape_params']['overlap'] != None :   
            im = break_the_image(transform_params['reshape_params']['stripes'], im, overlap=transform_params['reshape_params']['overlap'] );
    
    #for masks
    if transform_params['parts_alignment'] != None :
        if 'map' in transform_params['parts_alignment']:
            result.append(im)
            
        if 'layers' in transform_params['parts_alignment']:
            result.append(layersFromMap(im, transform_params['layers_sets']))
        
        return result;    
    
    #in other cases return one image
    result.append(im)
    return result;
def prepareDataset(image_path, transform_params):
    images = list()
    for i in xrange(len(image_path)):
        im = cv2.imread(image_path[i], cv2.IMREAD_COLOR)
        images_prepared = prepareImage(im, transform_params)
        
        images.append(images_prepared[0]) #only one image is supposed to return 
    return images

def unfold(im_from_base):
    im_transp = im_from_base.transpose((1,2,0))
    im = vstack([im_transp[:,:,:3], im_transp[:,:,3:6], im_transp[:,:,6:]])
    return cv2.cvtColor((im ).astype(np.uint8), cv2.COLOR_BGR2RGB)

def filterDataForLabels(labels_set, train_labels, train_camera_labels, train_images, image_names):
    
    filtered_labels =list()
    filtered_camera_labels =list()
    filtered_images = list()
    filtered_images_names = list()
    for i in xrange(len(train_labels)):
        if train_labels[i] in labels_set :
            filtered_labels.append(train_labels[i])
            filtered_camera_labels.append(train_camera_labels[i])
            filtered_images.append(train_images[i])
            filtered_images_names.append(image_names[i])
    return       filtered_labels, filtered_camera_labels, filtered_images, filtered_images_names
        

def create_db(path, prepared_images, labels):
    env = lmdb.open(path, map_size =107374182400L,  max_dbs=1) 
      
    key_counter = 0;
        
    with env.begin(write=True) as txn:
       
        for i in xrange(len(labels)):

            datum = caffe.io.array_to_datum(prepared_images[i],labels[i])
            txn.put(str(key_counter), datum.SerializeToString())
            key_counter = key_counter + 1;

    print "lmdb path ", path  
    
       

bounding_box_train = parseMarket1501(MARKET_ROOT + '/bounding_box_train')

train_labels = bounding_box_train[0]
train_camera_labels = bounding_box_train[1]
train_image_paths = bounding_box_train[2]
train_image_names = bounding_box_train[3]
#train_images


img_name_idx_dict = dict()
for i in xrange(len(train_image_names)):
    img_name_idx_dict[train_image_names[i]] = i


transform_params = dict()
transform_params['color_transformer'] = None
transform_params['reshape_params'] = dict()
transform_params['reshape_params']['stripes'] = 3
transform_params['reshape_params']['overlap'] = 10#11
transform_params['reshape_params']['resize'] = (60, 160)#(64, 128)
transform_params['reshape_params']['interpolation'] = None
transform_params['parts_alignment'] = None


train_images = prepareDataset(train_image_paths, transform_params)


# train_person_camera_image_dict = dict()
# train_person_image_dict = dict()
# for i in xrange(len(train_image_names)):
#     person, camera  = parse_market_1501_name(train_image_names[i])
#     if not (person, camera) in train_person_camera_image_dict :
#         train_person_camera_image_dict[(person, camera)] = list()
#         train_person_image_dict[person] = list()
#     train_person_camera_image_dict[(person, camera)].append(train_image_names[i])    
#     train_person_image_dict[person].append(train_image_names[i])    
        
    
# train_persons_for_eval = random.sample(train_person_image_dict.keys(), 100)

# train_query = []
# train_gallery = []
# for k in train_person_camera_image_dict:
#     if k[0] in train_persons_for_eval:
#         train_query.append(random.choice(train_person_camera_image_dict[k]))
#         train_gallery.extend(train_person_camera_image_dict[k])    


# train_labels_query = []
# train_camera_labels_query = []
# train_image_paths_query = []
# train_image_names_query = []
# train_images_query = []

# for i in xrange(len(train_query)):
#     idx = img_name_idx_dict[train_query[i]]
#     train_labels_query.append(train_labels[idx])
#     train_camera_labels_query.append(train_camera_labels[idx])
#     train_image_paths_query.append(train_image_paths[idx])
#     train_image_names_query.append(train_image_names[idx])
#     train_images_query.append(train_images[idx] )


# train_labels_gallery = []
# train_camera_labels_gallery  = []
# train_image_paths_gallery  = []
# train_image_names_gallery  = []
# train_images_gallery = []


# for i in xrange(len(train_gallery)):
#     idx = img_name_idx_dict[train_gallery[i]]
#     train_labels_gallery.append(train_labels[idx])
#     train_camera_labels_gallery.append(train_camera_labels[idx])
#     train_image_paths_gallery.append(train_image_paths[idx])
#     train_image_names_gallery.append(train_image_names[idx])
#     train_images_gallery.append(train_images[idx])
    

create_db(lmdb_train_path + '_tmp', train_images, train_labels)

def get_persons_list_db(db_path):
    env = lmdb.open(db_path, readonly=True) 
    print env.info(), env.max_key_size()
    persons = set();
    with env.begin() as txn:
        descriptors = list();

        cursor = txn.cursor()
        for key, value in cursor:
            datum = caffe_pb2.Datum.FromString(value)
            arr = caffe.io.datum_to_array(datum)
            label = datum.label;
            persons.add(label)
    return persons

def count(db_path):
    env = lmdb.open(db_path, readonly=True) 
    count=0
    with env.begin() as txn:
        descriptors = list();

        cursor = txn.cursor()
        for key, value in cursor:
            count+=1
    return count

bounding_box_test = parseMarket1501(MARKET_ROOT + '/bounding_box_test')

test_labels = bounding_box_test[0]
test_camera_labels = bounding_box_test[1]
test_image_paths = bounding_box_test[2]
test_image_names = bounding_box_test[3]

transform_params = dict()
transform_params['color_transformer'] = None
transform_params['reshape_params'] = dict()
transform_params['reshape_params']['stripes'] = 3
transform_params['reshape_params']['overlap'] = 10#11
transform_params['reshape_params']['resize'] = (60, 160)#(64, 128)
transform_params['reshape_params']['interpolation'] = None
transform_params['parts_alignment'] = None


test_images = prepareDataset(test_image_paths, transform_params)

query = parseMarket1501(MARKET_ROOT + '/query')

query_labels = query[0]
query_camera_labels = query[1]
query_image_paths = query[2]
query_image_names = query[3]

transform_params = dict()
transform_params['color_transformer'] = None
transform_params['reshape_params'] = dict()
transform_params['reshape_params']['stripes'] = 3
transform_params['reshape_params']['overlap'] = 10
transform_params['reshape_params']['resize'] = (60, 160)
transform_params['reshape_params']['interpolation'] = None
transform_params['parts_alignment'] = None


query_images = prepareDataset(query_image_paths, transform_params)

#Index calculation 

Index = dict()
Index['junk'] = set()
Index['distractor'] = set()
for name in test_image_names:
    if ifJunk(name):
        Index['junk'].add(name)
    elif ifDistractor(name):
        Index['distractor'].add(name)
for query in query_image_names:
    Index[query] = dict()
    Index[query]['pos'] = set()
    Index[query]['junk'] = set()
    
    person, camera = parse_market_1501_name(query)
    for name in test_image_names:
        if ifJunk(name) or ifDistractor(name):
            continue
        person_, camera_ = parse_market_1501_name(name)
        if person == person_ and camera !=camera_ :
            Index[query]['pos'].add(name)

        elif person == person_ and camera ==camera :
            Index[query]['junk'].add(name)
        

folder = '../market_experiment/'
MODEL_FILE = folder + 'protos/train_val_model.prototxt'
WEIGHTS = folder + '/snapshots/train_iter_150000.caffemodel'
caffe.set_mode_gpu()#net.set_phase_test()
net = caffe.Classifier(MODEL_FILE,WEIGHTS)



gescr_gallery= getLargeDescr(np.array(test_images), net, batch_size=50)
gescr_query = getLargeDescr(np.array(query_images), net, batch_size=50)
print len(gescr_gallery), len(test_images)
print len(gescr_query), len(query_images)
ranks,  img_names_sorted, places = ranking('cosine', gescr_query, query_image_names, gescr_gallery, test_image_names, 50, Index)

ranks_w= ranks / len(gescr_query) *1.0
print ranks_w

mAP_ = mAP(gescr_query, query_image_names, gescr_gallery, test_image_names, 50, Index)
print mAP_



