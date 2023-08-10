# a bit of setup as usual
import h5py, json

import numpy as np
import cPickle as pickle

from IPython.display import display, Image

h5_img_file_train = h5py.File('data/vqa_data_img_vgg_train.h5', 'r')
fv_im_train = h5_img_file_train.get('/images_train') # 82460 x 14 x 14 x 512

h5_img_file_test = h5py.File('data/vqa_data_img_vgg_test.h5', 'r')
fv_im_test = h5_img_file_test.get('/images_test') # 40504 x 14 x 14 x 512

h5_ques_file = h5py.File('data/vqa_data_prepro.h5', 'r')
ques_train = h5_ques_file.get('/ques_train') # 215375 x 26
ques_len_train = h5_ques_file.get('/ques_len_train') # 215375 x 1
img_pos_train = h5_ques_file.get('/img_pos_train') # 215375 x 1
ques_id_train = h5_ques_file.get('/ques_id_train') # 215375 x 1
answers = h5_ques_file.get('/answers') # 215375 x 1
split_train = h5_ques_file.get('/split_train') # 215375 x 1

ques_test = h5_ques_file.get('/ques_test') # 121512 x26
ques_len_test = h5_ques_file.get('/ques_len_test')
img_pos_test = h5_ques_file.get('/img_pos_test')
ques_id_test = h5_ques_file.get('/ques_id_test')
split_test = h5_ques_file.get('/split_test')
ans_test = h5_ques_file.get('/ans_test')

json_file = json.load(open('data/vqa_data_prepro.json', 'r'))
ix_to_word = json_file['ix_to_word']
ix_to_ans = json_file['ix_to_ans']

vocab_size = len(ix_to_word) # 12604

num_samples = 8000

qa_data_train_small = []
train_im_small_idx = []

for ix in xrange(num_samples * 3):
    qa_data_train_small.append((ques_train[ix], ques_len_train[ix], ix / 3, answers[ix]))
    if ix % 3 == 0:
        train_im_small_idx.append(img_pos_train[ix])

train_im_small = []
for im_ix in train_im_small_idx:
    train_im_small.append(fv_im_train[im_ix, :])

with open('data/qa_data_train_small.pkl', 'wb') as fp:
    pickle.dump(qa_data_train_small, fp)

with h5py.File('data/vqa_data_img_vgg_train_small.h5', 'w') as hf:
    hf.create_dataset('images_train', data=train_im_small)

num_samples = 4000

qa_data_test_small = []
test_im_small_idx = []

for ix in xrange(num_samples * 3):
    qa_data_test_small.append((ques_test[ix], ques_len_test[ix], ix / 3, ans_test[ix]))
    if ix % 3 == 0:
        test_im_small_idx.append(img_pos_test[ix])

test_im_small = []
for im_ix in test_im_small_idx:
    test_im_small.append(fv_im_test[im_ix, :])

with open('data/qa_data_test_small.pkl', 'wb') as fp:
    pickle.dump(qa_data_test_small, fp)

with h5py.File('data/vqa_data_img_vgg_test_small.h5', 'w') as hf:
    hf.create_dataset('images_test', data=test_im_small)

num_samples = 100

qa_data_train_small = []
train_im_small_idx = []

for ix in xrange(num_samples * 3):
    qa_data_train_small.append((ques_train[ix], ques_len_train[ix], ix / 3, answers[ix]))
    if ix % 3 == 0:
        train_im_small_idx.append(img_pos_train[ix])

train_im_small = []
for im_ix in train_im_small_idx:
    train_im_small.append(fv_im_train[im_ix, :])
    
with open('data/qa_data_train_tiny.pkl', 'wb') as fp:
    pickle.dump(qa_data_train_small, fp)

with h5py.File('data/vqa_data_img_vgg_train_tiny.h5', 'w') as hf:
    hf.create_dataset('images_train', data=train_im_small)

num_samples = 100

qa_data_test_small = []
test_im_small_idx = []

for ix in xrange(num_samples * 3):
    qa_data_test_small.append((ques_test[ix], ques_len_test[ix], ix / 3, ans_test[ix]))
    if ix % 3 == 0:
        test_im_small_idx.append(img_pos_test[ix])

test_im_small = []
for im_ix in test_im_small_idx:
    test_im_small.append(fv_im_test[im_ix, :])
    
with open('data/qa_data_test_tiny.pkl', 'wb') as fp:
    pickle.dump(qa_data_test_small, fp)

with h5py.File('data/vqa_data_img_vgg_test_tiny.h5', 'w') as hf:
    hf.create_dataset('images_test', data=test_im_small)

ques, ques_len, im_ix, ans = zip(*qa_data_test_small)
print [ix_to_word.get(str(ix), 'UNK') for ix in ques[0]], im_ix[0], img_pos_test[0], ix_to_ans.get(str(ans[0]), 'UNK')

raw_test_json = json.load(open('data/vqa_raw_test.json', 'r'))

for i in xrange(0, 30, 3):
    print "Q: %s - GT: %s" % (raw_test_json[i]['question'], raw_test_json[i]['ans'])
    print "Q: %s - GT: %s" % (raw_test_json[i+1]['question'], raw_test_json[i+1]['ans'])
    print "Q: %s - GT: %s" % (raw_test_json[i+2]['question'], raw_test_json[i+2]['ans'])
    display(Image(filename='data/imgs/%d.jpg' % (i/3 + 1)))

