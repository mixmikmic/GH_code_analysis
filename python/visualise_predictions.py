from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

import os
import numpy as np
import time
from io import open
import math
import cPickle as pkl
import yaml

os.environ["THEANO_FLAGS"] = "floatX=float32,device=gpu0"
import theano

from model import Model
from utils import get_vocabulary, dump_params, load_params, iterate_pkl_minibatches, iterate_minibatches
import tables
from sklearn.metrics.pairwise import cosine_similarity

class Tester:
    def __init__(self, config, shared_theano_params=None):
        self.config = config
        if config['exp_id'] != '':
            # helps us keep the parameters in different spaces
            # instead of only in the same model_name file
            config['model_name'] = '{}-{}'.format(config['exp_id'], config['model_name'])
        self.train_path = config['train']
        self.fc7_path = config['fc7_train']
        self.val_path = config['val']
        self.val_fc7_path = config['fc7_val']
        self.model = Model(config, load=True)
        self.update_count = 0
        self.batch_size = config['batch_size']
        self.use_dropout = config['dropout']
        self.verbose = config['verbose']

    def get_val_data_iterator(self):
        """
        Returns an iterator over validation data.

        Returns: iterator
        """
        if self.val_path.endswith('pkl'):
            return iterate_pkl_minibatches(self.val_path, self.val_fc7_path, shuffle=False,
                                       batch_size=self.batch_size)
        else:
            return iterate_minibatches(self.val_path, self.val_fc7_path, shuffle=False,
                                       batch_size=self.batch_size)

    def get_predictions(self):
        # Collect the predicted image vectors on the validation dataset
        # We do this is batches so a dataset with a large validation split
        # won't cause GPU OutOfMemory errors.
        all_preds = None
        for sentences in self.get_val_data_iterator():
            x, x_mask, y = self.model.prepare_batch(sentences[0], sentences[1])
            predictions = self.model.predict_on_batch(x, x_mask)
            if all_preds == None:
                all_preds = predictions
            else:
                all_preds = np.vstack((all_preds, predictions))
        return all_preds

    def calculate_ranking(self, predictions, fc7_path, k=1, npts=None):
        """
        :param predictions: matrix of predicted image vectors
        :param fc7_path: path to the true image vectors
        :param k: number of predictions per image (usually based on the number
        of sentence encodings)

        TODO: vectorise the calculation
        """

        # Normalise the predicted vectors
        for i in range(len(predictions)):
            predictions[i] /= np.linalg.norm(predictions[i])

        fc7_file = tables.open_file(fc7_path, mode='r')
        fc7_vectors = fc7_file.root.feats[:]
        images = fc7_vectors[:]

        # Normalise the true vectors
        for i in range(len(images)):
            images[i] /= np.linalg.norm(images[i])

        if npts == None:
            npts = predictions.shape[0]
            if npts > 25000:
                # The COCO validation pkl contains 25,010 instances???
                npts = 25000

        ranks = np.full(len(images), 1e20)
        for index in range(npts):
            # Get the predicted image vector
            p = predictions[index]

            # Compute cosine similarity between predicted vector and the
            # true vectors
            sim = np.dot(p, images.T)
            inds = np.argsort(sim) # should we reverse list?

            # Score
            # Iterate through the possible trues
            target = int(math.floor(index/k))
            tmp = np.where(inds == target)
            #print("Index {} target {} tmp {}".format(index, target, tmp[0]))
            tmp = tmp[0][0]
            if tmp < ranks[target]:
                ranks[target] = tmp

        # Compute metrics
        r1, r5, r10, medr = self.ranking_results(ranks)
        return (r1, r5, r10, medr)

    def ranking_results(self, ranks):
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        logger.info('R@1 {} R@5 {} R@10 {} Median {}'.format(r1, r5, r10, medr))
        return r1, r5, r10, medr

    def predict_images(self):
        """ 
        Predict the most likely images, given sentences 
        """

        logger.info('Imagineting the dev set')
        start_time = time.time()

        # Collect the predicted image vectors on the validation dataset
        # We do this is batches so a dataset with a large validation split
        # won't cause GPU OutOfMemory errors.
        all_preds = self.get_predictions()

        # Measure the ranking performance on the validation dataset
        self.calculate_ranking(all_preds, self.config['fc7_val'], k=self.config['ranking_k'])

    def test(self):
        self.predict_images()

def build_test(config):
    """
    Make predictions
    :param config: yaml-file with configuration
    :return:
    """

    # print config
    logger.info('Testing with the following (loaded) config:')
    for arg, value in config.items():
        logger.info('{}: {}'.format(arg, value))
    
    # load model
    tester = Tester(config)
    return tester

#os.chdir('src/imaginet')
config_file = "configs/multi30k.inceptionv3.yaml"
config = yaml.load(open(config_file, mode='rb'))
config['exp_id'] = ''
config['word_vocabulary'] = '../nmt/data/wmt_task1/en_dict.json'
config['fc7_val'] = '../' + config['fc7_val']
config['val'] = '../' + config['val']
# Insane hack to get the correct path
config['model_name'] = '../../models/backup/run4-inceptionv3-w620-h1000-meanmlp-tanh-constrastive0.1-adam-1e-4-dropout_hid0.3-decay_c1e-8-alltoks.npz.29.medr11.0'
image_path = '/home/delliott/data/flickr30k/'
image_list_file = '../nmt/data/wmt_task1/val_images.txt'
with open(image_list_file) as f:
    image_list = [line.strip() for line in f]

def get_sentence(filename, index):
     with open(filename, mode='r', encoding='utf-8') as f:
        data = []
        i = 0
        for line in f:
            data.append((line.replace('\n',''), i))
            i += 1
        return data[index]
    
def calculate_top5(predictions, fc7_path, index, k=1, npts=None):
        """
        :param predictions: matrix of predicted image vectors
        :param fc7_path: path to the true image vectors
        :param k: number of predictions per image (usually based on the number
        of sentence encodings)

        TODO: vectorise the calculation
        """

        # Normalise the predicted vectors
        for i in range(len(predictions)):
            predictions[i] /= np.linalg.norm(predictions[i])

        fc7_file = tables.open_file(fc7_path, mode='r')
        fc7_vectors = fc7_file.root.feats[:]
        images = fc7_vectors[:]

        # Normalise the true vectors
        for i in range(len(images)):
            images[i] /= np.linalg.norm(images[i])

        if npts == None:
            npts = predictions.shape[0]
            if npts > 25000:
                # The COCO validation pkl contains 25,010 instances???
                npts = 25000

        # Get the predicted image vector
        p = predictions[index]

        # Compute cosine similarity between predicted vector and the
        # true vectors
        sim = np.dot(p, images.T)
        inds = np.argsort(sim) # should we reverse list?
        return inds

tester = build_test(config)

tester.test()

preds = tester.get_predictions()

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import skimage
import skimage.transform
import skimage.io
# keep aspect ratio, and center crop
def LoadImage(file_name, resize=256, crop=256):
  image = Image.open(file_name)
  width, height = image.size

  if width > height:
    width = (width * resize) / height
    height = resize
  else:
    height = (height * resize) / width
    width = resize
  left = (width  - crop) / 2
  top  = (height - crop) / 2
  image_resized = image.resize((int(width), int(height)), Image.BICUBIC).crop((left, top, left + crop, top + crop))
  data = np.array(image_resized.convert('RGB').getdata()).reshape(crop, crop, 3)
  data = data.astype('float32') / 255
  return data

idx = np.random.randint(0, len(preds)) # random image
sentence = get_sentence(config['val'], idx)[0] # groundtruth sentence
img = LoadImage(image_path+image_list[idx])
top5 = calculate_top5(preds, config['fc7_val'], idx)

fig = plt.figure(1, figsize = (20,10))

for ii in xrange(5):
    ax = fig.add_subplot(1, 5, ii+1)
    pred_image = LoadImage(image_path+image_list[top5[ii]])
    ax.imshow(pred_image)
    ax.axis('off')
                     
plt.show()
print("{}: {}".format(idx, sentence))

ranks = []
for x in xrange(len(preds)):
    idx = x # random image
    sentence = get_sentence(config['val'], idx)[0] # groundtruth sentence
    img = LoadImage(image_path+image_list[idx])
    top5 = calculate_top5(preds, config['fc7_val'], idx)
    ranks.append(np.where(top5 == idx)[0][0])

for y in ranks:
    print(y)



