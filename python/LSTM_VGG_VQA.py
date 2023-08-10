# don't re-inventing the wheel
import h5py, json, spacy

import numpy as np
import cPickle as pickle

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from model import LSTMModel
from utils import prepare_ques_batch, prepare_im_batch

# run `python -m spacy.en.download` to collect the embeddings (1st time only)
embeddings = spacy.en.English()
word_dim = 300

h5_img_file_tiny = h5py.File('data/vqa_data_img_vgg_train_tiny.h5', 'r')
fv_im_tiny = h5_img_file_tiny.get('/images_train')

with open('data/qa_data_train_tiny.pkl', 'rb') as fp:
    qa_data_tiny = pickle.load(fp)

json_file = json.load(open('data/vqa_data_prepro.json', 'r'))
ix_to_word = json_file['ix_to_word']
ix_to_ans = json_file['ix_to_ans']

vocab_size = len(ix_to_word)
print "Loading tiny dataset of %d image features and %d question/answer pairs for training." % (len(fv_im_tiny), len(qa_data_tiny)) 

questions, ques_len, im_ix, ans = zip(*qa_data_tiny)

nb_classes = 1000
max_ques_len = 26

X_ques = prepare_ques_batch(questions, ques_len, max_ques_len, embeddings, word_dim, ix_to_word)
X_im = prepare_im_batch(fv_im_tiny, im_ix)
y = np.zeros((len(ans), nb_classes))
y[np.arange(len(ans)), ans] = 1

model = LSTMModel()
model.build()

loss = model.fit(X_ques, X_im, y, nb_epoch=50, batch_size=50)

plt.plot(loss.history['loss'], label='train_loss')
plt.plot(loss.history['acc'], label='train_acc')
plt.legend(loc='best')

h5_img_file_test_tiny = h5py.File('data/vqa_data_img_vgg_test_tiny.h5', 'r')
fv_im_test_tiny = h5_img_file_test_tiny.get('/images_test')

with open('data/qa_data_test_tiny.pkl', 'rb') as fp:
    qa_data_test_tiny = pickle.load(fp)
    
print "Loading tiny dataset of %d image features and %d question/answer pairs for testing" % (len(fv_im_test_tiny), len(qa_data_test_tiny)) 

questions, ques_len, im_ix, ans = zip(*qa_data_test_tiny)

X_ques_test = prepare_ques_batch(questions, ques_len, max_ques_len, embeddings, word_dim, ix_to_word)
X_im_test = prepare_im_batch(fv_im_test_tiny, im_ix)
y_test = np.zeros((len(ans), nb_classes))
y_test[np.arange(len(ans)), [494 if a > 1000 else a for a in ans]] = 1

loss, acc = model.evaluate(X_ques_test, X_im_test, y_test)

print loss, acc

