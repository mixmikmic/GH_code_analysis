import vgg19
import numpy as np
from urllib import urlretrieve
import gzip
import os
import matplotlib.pyplot as plt
import skimage.transform
from lasagne.utils import floatX


get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 1')

import glob

def ensure_data_exists ():
    """ 
    Loads CPPN adversarial examples if they don't already exist.
    """
    filename = "fooling_images_5000_cppn.tar.gz"
    url = "http://www.evolvingai.org/share/fooling_images_5000_cppn.tar.gz"
    
    if not os.path.exists('../datasets/{0}'.format(filename)):
        print "Adversarial data set didn't exist, downloading..."
        urlretrieve(url, '../datasets/'+filename)
        print "Download complete."
        print 'unzipping...'
        get_ipython().system('tar -xzvf ../datasets/fooling_images_5000_cppn.tar.gz')
    else:
        print "found it!"
    

def read_image (img_file, mean_image):
#     try:
        im = plt.imread(img_file, '.png')

        h, w, _ = im.shape
        if h < w:
            im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
        else:
            im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)
            
        # Central crop to 224x224
        h, w, _ = im.shape
        im = im[h//2-112:h//2+112, w//2-112:w//2+112]
        rawim = np.copy(im).astype('uint8')

        # Shuffle axes to c01
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
        # Convert to BGR
        im = im[::-1, :, :]
        im = im - mean_image[:,None,None]
        print im.shape
        return rawim, floatX(im[None, :, :, :])
    

#     except:
#         # Abort
#         print "skipping link " + img_file
#         return None, np.zeros((1,))
    
def load_data (mean_image):
    """ 
    Import the evolutionary adversarial examples.  
    
    Returns:
    :images: a (N, C, H, W) tensor containing the validation dataset.
    :metadata: a list containing file metadata corresponding to each image.   
    """
    
    

    images = None
    metadata = []
    raw_im = np.zeros ((5000, 224, 224, 3), dtype=np.uint8)
    images = np.zeros ((5000, 3, 224, 224), dtype=np.float32)
    print images.shape
    i = 0
    
    for filename in glob.glob('../datasets/fooling/*.png'):

        print i, "looking at ", filename
        rawim, result = read_image (filename, mean_image)
        if result.any():
            images[i, :, :, :] = result 
            raw_im[i, :, :, :] = rawim[None, :, :, :] 
            i += 1 
        # read metadata
        file_info = filename.split('_')
        file_info[-1][:-4]
        data = {}

        data['index'] = i
        data['name'] = filename
        data['label'] = file_info[-2]
        data['confidence'] = float(file_info[-1][:-4]) # class confidence
        metadata.append(data)
        
    return raw_im, images, metadata


#ensure_data_exists()
import vgg19
import lasagne

net = vgg19.build_model()
classes, mean_image, values = vgg19.load_weights()
lasagne.layers.set_all_param_values(net['prob'], values)

raw_im, images, metadata = load_data(mean_image)

from time import time

N = 9
get_ipython().magic('matplotlib inline')
# test metadata

# store the predictions of vgg
predictions = []
QUANTA = 10


for j in np.arange(2000, 2000+N, QUANTA):
    t0 = time()
    sample = images[j:j+QUANTA, :, :, :]
    prob = np.array(lasagne.layers.get_output(net['prob'], 
                    sample, deterministic=True).eval())
    t1 = time()
    print "ran ex. {0} to {1} in {2} seconds".format(j, j + QUANTA, t1 - t0)
    y_predict = np.argmax(prob ,axis=1)
    print y_predict.shape, y_predict
    
    for i in xrange (QUANTA):
        res = {}
        res['label'] = classes[y_predict[i]]
        res['index'] = j + i
        res['confidence'] = prob[i, y_predict[i]]
        predictions.append(res)
    
    
# # run the VGG net

# #short_images = images[:10, :, :, :]

# #prob = np.array(lasagne.layers.get_output(net['prob'], short_images, deterministic=True).eval())

# for i in xrange(10):
#     print images[i].shape
#     plt.imshow(raw_im[i])
#     plt.axis('off')
#     plt.text(250, 70, 'Guess: {0} \nConfidence: {1}'.format(classes[ int(metadata[i]['label']) ], 
#                                                                metadata[i]['confidence']), fontsize=14)
#     plt.show()

print predictions

# load the prediction data
import cPickle as pickle
with open('../datasets/cppn_results', 'r') as f:
    predictions = pickle.load(f)

# save the data

import cPickle as pickle

with open('../datasets/cppn_results', 'w+') as f:
    pickle.dump(predictions, f)

# Sort and Check status of data

predictions.sort(key=(lambda p: -metadata[p['index']]['confidence']))

print predictions
#print predictions


# Run some statistics


highest_conf = max(predictions, key=lambda k:k['confidence'])
print highest_conf

# label variance

from collections import Counter

pred_frequency = Counter()
for elt in predictions:
    pred_frequency[ elt['label'] ] += 1

print pred_frequency

# print a random sample

num_examples = 10

indices = np.arange(2000, 2010)

for i in xrange(10):
    plt.imshow(raw_im[i+2000])
    plt.axis('off')
    plt.text(250, 70, 'AlexNet Guess: {0} \nConfidence: {1} \n\n VGG Guess: {2}\n Confidence: {3}'
             .format(classes[ int(metadata[i+2000]['label']) ], 
                     metadata[i+2000]['confidence'], predictions[i]['label'], predictions[i]['confidence']), 
                     fontsize=14)
    plt.show()

get_ipython().magic('matplotlib inline')
num_examples = 10

indices = np.arange(2000, 2010)

plt.imshow(raw_im[2000])

f, plot = plt.subplots(1, 10, figsize=(15, 4))
plt.axis('off')

count = 0
for index in indices:
    
    x, y = count / 5, count % 5
    
    print index
    i = index - 2000
    
    label = classes[ int(metadata[index]['label'] )]
    if  ',' in label:
        shortened_label = label[:label.index(',')]
    else:
        shortened_label = label
    shortened_label = '\n'.join(shortened_label.split(' '))
    
    plot[count].text(0, -25, "{0:.1f}% \n{1}".format(100 * metadata[index]['confidence'], 
                     shortened_label),
                     fontsize=14)
    plot[count].imshow(raw_im[index])
    plot[count].axis('off')
    
    
    vgg_label = ',\n'.join(predictions[i]['label'].split(','))
    
    plot[count].text(0, 350, "{0:.1f}% \n{1}".format(100 * predictions[i]['confidence'], 
                     vgg_label),
                     fontsize=12)
    
    count += 1
#     plt.axis('off')
#     plt.text(250, 70, 'AlexNet Guess: {0} \nConfidence: {1} \n\n VGG Guess: {2}\n Confidence: {3}'
#              .format(classes[ int(metadata[i]['label']) ], 
#                      metadata[i]['confidence'], predictions[i]['label'], predictions[i]['confidence']), 
#                      fontsize=14)
   # plt.show()
plt.tight_layout()
plt.savefig('adversarial.png', dpi=300)
plt.show()





