import json
import os, os.path
import pickle

train_val = json.load(open('videodatainfo_2017.json', 'r'))


# combine all images and annotations together
sentences = train_val['sentences']

# for efficiency lets group annotations by video
itoa = {}
for s in sentences:
    videoid_buf = s['video_id']
    videoid = int(videoid_buf[5:])
    if not videoid in itoa: itoa[videoid] = []
    itoa[videoid].append(s)
    
output = open('./DATA/word_features/captions.pkl', 'wb')
pickle.dump(itoa, output)
output.close()

import numpy as np

"""Functions to do the following:
            * Create vocabulary
            * Create dictionary mapping from word to word_id
            * Map words in captions to word_ids"""

def build_vocab(word_count_thresh):
    """Function to create vocabulary based on word count threshold.
        Input:
                word_count_thresh: Threshold to choose words to include to the vocabulary
        Output:
                vocabulary: Set of words in the vocabulary"""
    
    pkl_file = open('./DATA/word_features/captions.pkl', 'rb')
    sentences = pickle.load(pkl_file)
    pkl_file.close()

    unk_required = False
    all_captions = []
    word_counts = {}
    for vid in sentences.keys():
        for cid in range(0,20):
            caption = sentences[vid][cid]['caption']
            caption = '<BOS> ' + caption + ' <EOS>'
            all_captions.append(caption)
            for word in caption.split(' '):
                if word in word_counts.keys():
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
    for word in word_counts.keys():
        if word_counts[word] < word_count_thresh:
            word_counts.pop(word)
            unk_required = True
    return word_counts,unk_required

def word_to_word_ids(word_counts,unk_required, vocab_size):
    """Function to map individual words to their id's.
        Input:
                word_counts: Dictionary with words mapped to their counts
        Output:
                word_to_id: Dictionary with words mapped to their id's. 
    """

    count = 0
    word_to_id = {}
    id_to_word = {}

    # Taking the most frequent vocab_size words
    words = [word for word in word_counts.keys()]
    values = [word_counts[word] for word in words]
    sorted_indices = np.argsort(values)
    words = np.array(words)
    most_freq_words = words[sorted_indices[::-1][0:vocab_size]]
    
    id_to_word = [most_freq_words[i] for i in range(most_freq_words.shape[0])] 
    
    #word2idx
    word_to_id = {}
    for i in range(len(id_to_word)):
        word_to_id[id_to_word[i]] = i
    
    print(word_to_id['<EOS>'])
    index = word_to_id['<EOS>']
    word = id_to_word[0]
    print(index,word)
    
    word_to_id['<EOS>'] = 0
    id_to_word[0] = '<EOS>'
    word_to_id[word] = index
    id_to_word[index] = word
    
    return word_to_id,id_to_word

def convert_caption(caption,word_to_id,max_caption_length):
    """Function to map each word in a caption to it's respective id and to retrieve caption masks
        Input:
                caption: Caption to convert to word_to_word_ids
                word_to_id: Dictionary mapping words to their respective id's
                max_caption_length: Maximum number of words allowed in a caption
        Output:
                caps: Captions with words mapped to word id's
                cap_masks: Caption masks with 1's at positions of words and 0's at pad locations"""
    caps,cap_masks = [],[]
    if type(caption) == 'str':
        caption = [caption] # if single caption, make it a list of captions of length one
    for cap in caption:
        cap = '<BOS> '+cap+' <EOS>'
        nWords = cap.count(' ') + 1
        if nWords >= max_caption_length:
            carr = cap.split(' ')
            carr = carr[0:(max_caption_length-2)]
            cap  = ' '.join(carr)
            cap  = cap + ' <EOS>'
            nWords = cap.count(' ')+1
        cap = cap + ' <EOS>'*(max_caption_length-nWords)
        cap_masks.append([1.0]*nWords + [0.0]*(max_caption_length-nWords))
        curr_cap = []
        for word in cap.split(' '):
            #print(word)
            if word in word_to_id.keys():
                curr_cap.append(word_to_id[word]) # word is present in chosen vocabulary
            else:
                curr_cap.append(word_to_id['<UNK>']) # word not present in chosen vocabulary
        caps.append(curr_cap)
        #print('Caption_Length:',len(caps[0]))
    return np.array(caps),np.array(cap_masks)

## Get the list of the files we have extracted features
import os
from sklearn.model_selection import train_test_split

video_list = os.listdir('./DATA/features')
videos = []
for item in video_list:
    videos.append(item.split('-')[0])

video_train, video_test = train_test_split(videos, test_size=0.1, random_state=42)
video_train, video_val = train_test_split(video_train, test_size=0.1, random_state=42)

print('Training Videos -', len(video_train))
print('Testing Videos -', len(video_test))
print('Validation Videos -', len(video_val))

import numpy as np
import tensorflow as tf
import glob
import cv2
import imageio
import pickle
np.random.seed(0)
#Global initializations
n_lstm_steps = 30
DATA_DIR = './DATA/'
VIDEO_DIR = DATA_DIR + 'features/'
YOUTUBE_CLIPS_DIR = DATA_DIR + 'videos/'
TEXT_DIR = DATA_DIR+'word_features/'
pkl_file = open('./DATA/word_features/captions.pkl', 'rb')
sentences = pickle.load(pkl_file)
pkl_file.close()
word_counts,unk_required = build_vocab(0)
word2id,id2word = word_to_word_ids(word_counts,unk_required, len(word_counts.keys()))
video_files = video_train
val_files = video_val

print ("{0} files processed".format(len(video_files)))

def get_bias_vector():
    """Function to return the initialization for the bias vector
       for mapping from hidden_dim to vocab_size.
       Borrowed from neuraltalk by Andrej Karpathy"""
    bias_init_vector = np.array([1.0*word_counts[id2word[i]] for i in range(len(id2word))])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)
    return bias_init_vector

def fetch_data_batch(batch_size):
    """Function to fetch a batch of video features, captions and caption masks
        Input:
                batch_size: Size of batch to load
        Output:
                curr_vids: Features of the randomly selected batch of video_files
                curr_caps: Ground truth (padded) captions for the selected videos
                curr_masks: Mask for the pad locations in curr_caps"""
    curr_batch_vids = np.random.choice(video_files,batch_size)
    curr_vids = np.array([[np.mean(np.load(VIDEO_DIR + vid+'-30-features' + '.npy'), axis=0) for i in range(30)] for vid in curr_batch_vids])
    captions = [np.random.choice(sentences[int(vid[5:])],1)[0]['caption'] for vid in curr_batch_vids]
    curr_caps,curr_masks = convert_caption(captions,word2id,n_lstm_steps)
    return curr_vids,curr_caps,curr_masks

def fetch_data_batch_val(batch_size):
    """Function to fetch a batch of video features from the validation set and its captions.
        Input:
                batch_size: Size of batch to load
        Output:
                curr_vids: Features of the randomly selected batch of video_files
                curr_caps: Ground truth (padded) captions for the selected videos"""

    curr_batch_vids = np.random.choice(val_files,batch_size)
    curr_vids = np.array([[np.mean(np.load(VIDEO_DIR + vid+'-30-features' + '.npy'), axis=0) for i in range(30)] for vid in curr_batch_vids])
    captions = [np.random.choice(sentences[int(vid[5:])],1)[0]['caption'] for vid in curr_batch_vids]
    curr_caps,curr_masks = convert_caption(captions,word2id,n_lstm_steps)
    return curr_vids,curr_caps,curr_masks, curr_batch_vids


def print_in_english(caption_idx):
    """Function to take a list of captions with words mapped to ids and
        print the captions after mapping word indices back to words."""
    captions_english = [[id2word[word] for word in caption] for caption in caption_idx]
    for i,caption in enumerate(captions_english):
        if '<EOS>' in caption:
            caption = caption[0:caption.index('<EOS>')]
        print (str(i+1) + ' ' + ' '.join(caption))
        print ('..................................................')

def playVideo(video_urls):
    video = imageio.get_reader(YOUTUBE_CLIPS_DIR + video_urls[0] + '.mp4','ffmpeg')
    for frame in video:
        fr = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow('frame',fr)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

a = np.mean(np.load('./DATA/features/video4804'+'-30-features' + '.npy'), axis=0)
a.shape

a = (np.load('./DATA/features/video4804'+'-30-features' + '.npy'))
a.shape

print(len(word2id))

tmp_val = 'video3707'
np.random.choice(sentences[int(tmp_val[5:])],1)[0]['caption']

tdata = np.load(VIDEO_DIR+'video0-30-features.npy')
tdata.shape

len(word_counts.keys())

print(id2word[0], word2id['a'])

import numpy as np
import tensorflow as tf
import sys
#GLOBAL VARIABLE INITIALIZATIONS TO BUILD MODEL
n_steps = 30
hidden_dim = 500
frame_dim = 2048
batch_size = 1
vocab_size = len(word2id)
bias_init_vector = get_bias_vector()
n_steps_vocab = 30

def build_model():
    """This function creates weight matrices that transform:
            * frames to caption dimension
            * hidden state to vocabulary dimension
            * creates word embedding matrix """

    print ("Network config: \nN_Steps: {}\nHidden_dim:{}\nFrame_dim:{}\nBatch_size:{}\nVocab_size:{}\n".format(n_steps,
                                                                                                    hidden_dim,
                                                                                                    frame_dim,
                                                                                                    batch_size,
                                                                                                    vocab_size))

    #Create placeholders for holding a batch of videos, captions and caption masks
    video = tf.placeholder(tf.float32,shape=[batch_size,n_steps,frame_dim],name='Input_Video')
    caption = tf.placeholder(tf.int32,shape=[batch_size,n_steps_vocab],name='GT_Caption')
    caption_mask = tf.placeholder(tf.float32,shape=[batch_size,n_steps_vocab],name='Caption_Mask')
    dropout_prob = tf.placeholder(tf.float32,name='Dropout_Keep_Probability')

    with tf.variable_scope('Im2Cap') as scope:
        W_im2cap = tf.get_variable(name='W_im2cap',shape=[frame_dim,
                                                    hidden_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
        b_im2cap = tf.get_variable(name='b_im2cap',shape=[hidden_dim],
                                                    initializer=tf.constant_initializer(0.0))
    with tf.variable_scope('Hid2Vocab') as scope:
        W_H2vocab = tf.get_variable(name='W_H2vocab',shape=[hidden_dim,vocab_size],
                                                         initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
        b_H2vocab = tf.Variable(name='b_H2vocab',initial_value=bias_init_vector.astype(np.float32))

    with tf.variable_scope('Word_Vectors') as scope:
        word_emb = tf.get_variable(name='Word_embedding',shape=[vocab_size,hidden_dim],
                                                                initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
    print ("Created weights")

    #Build two LSTMs, one for processing the video and another for generating the caption
    with tf.variable_scope('LSTM_Video',reuse=None) as scope:
        lstm_vid = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
        lstm_vid = tf.nn.rnn_cell.DropoutWrapper(lstm_vid,output_keep_prob=dropout_prob)
    with tf.variable_scope('LSTM_Caption',reuse=None) as scope:
        lstm_cap = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
        lstm_cap = tf.nn.rnn_cell.DropoutWrapper(lstm_cap,output_keep_prob=dropout_prob)

    #Prepare input for lstm_video
    video_rshp = tf.reshape(video,[-1,frame_dim])
    video_rshp = tf.nn.dropout(video_rshp,keep_prob=dropout_prob)
    video_emb = tf.nn.xw_plus_b(video_rshp,W_im2cap,b_im2cap)
    video_emb = tf.reshape(video_emb,[batch_size,n_steps,hidden_dim])
    padding = tf.zeros([batch_size,n_steps-1,hidden_dim])
    video_input = tf.concat([video_emb,padding],1)
    #video_input=video_emb
    print ("Video_input: {}".format(video_input.get_shape()))
    #Run lstm_vid for 2*n_steps-1 timesteps
    with tf.variable_scope('LSTM_Video') as scope:
        out_vid,state_vid = tf.nn.dynamic_rnn(lstm_vid,video_input,dtype=tf.float32)
    print ("Video_output: {}".format(out_vid.get_shape()))

    #Prepare input for lstm_cap
    padding = tf.zeros([batch_size,n_steps_vocab,hidden_dim])
    caption_vectors = tf.nn.embedding_lookup(word_emb,caption[:,0:n_steps_vocab-1])
    caption_vectors = tf.nn.dropout(caption_vectors,keep_prob=dropout_prob)
    caption_2n = tf.concat([padding,caption_vectors],1)
    #caption_2n = caption_vectors
    caption_input = tf.concat([caption_2n,out_vid],2)
    print ("Caption_input: {}".format(caption_input.get_shape()))
    #Run lstm_cap for 2*n_steps-1 timesteps
    with tf.variable_scope('LSTM_Caption') as scope:
        out_cap,state_cap = tf.nn.dynamic_rnn(lstm_cap,caption_input,dtype=tf.float32)
    print ("Caption_output: {}".format(out_cap.get_shape()))

    #Compute masked loss
    output_captions = out_cap[:,n_steps_vocab:,:]
    output_logits = tf.reshape(output_captions,[-1,hidden_dim])
    output_logits = tf.nn.dropout(output_logits,keep_prob=dropout_prob)
    output_logits = tf.nn.xw_plus_b(output_logits,W_H2vocab,b_H2vocab)
    output_labels = tf.reshape(caption[:,1:],[-1])
    caption_mask_out = tf.reshape(caption_mask[:,1:],[-1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_logits,labels=output_labels)
    masked_loss = loss*caption_mask_out
    loss = tf.reduce_sum(masked_loss)/tf.reduce_sum(caption_mask_out)
    return video,caption,caption_mask,output_logits,loss,dropout_prob

db1 = None
db2 = None
db3 = None
def train():
    global db1,db2,db3
    with tf.Graph().as_default():
        learning_rate = 0.0001
        video,caption,caption_mask,output_logits,loss,dropout_prob = build_model()
        optim = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
        nEpoch = 300
        nIter = int(nEpoch*6000/batch_size)
        
        ckpt_file = './checkpoint_2d/model_58000.ckpt.meta'

        saver = tf.train.Saver()
        with tf.Session() as sess:
            if ckpt_file:
                saver_ = tf.train.import_meta_graph(ckpt_file)
                saver_.restore(sess,'./checkpoint_2d/model_58000.ckpt')
                print ("Restored model")
            else:
                sess.run(tf.global_variables_initializer())
            for i in range(nIter):
                #print(i)
                vids,caps,caps_mask = fetch_data_batch(batch_size=batch_size)
                db1,db2,db3 = vids, caps, caps_mask
                #print(type(vids),type(caps), type(caps_mask))
                #print(vids,caps, caps_mask)
                _,curr_loss,o_l = sess.run([optim,loss,output_logits],feed_dict={video:vids,
                                                                            caption:caps,
                                                                            caption_mask:caps_mask,
                                                                            dropout_prob:0.5})

                if i%1000 == 0:
                    print ("\nIteration {} \n".format(i))
                    out_logits = o_l.reshape([batch_size,n_steps_vocab-1,vocab_size])
                    output_captions = np.argmax(out_logits,2)
                    #print_in_english(output_captions[0:4])
                    #print ("GT Captions")
                    #print_in_english(caps[0:4])
                    print ("Current train loss: {} ".format(curr_loss))
                    vids,caps,caps_mask,_ = fetch_data_batch_val(batch_size=batch_size)
                    db1,db2,db3 = vids,caps,caps_mask
                    curr_loss,o_l = sess.run([loss,output_logits],feed_dict={video:vids,
                                                                            caption:caps,
                                                                            caption_mask:caps_mask,
                                                                            dropout_prob:1.0})
                    out_logits = o_l.reshape([batch_size,n_steps_vocab-1,vocab_size])
                    output_captions = np.argmax(out_logits,2)
                    print_in_english(output_captions[0:2])
                    print ("GT Captions")
                    print_in_english(caps[0:2])
                    print ("Current validation loss: {} ".format(curr_loss))

                if i%2000 == 0:
                    saver.save(sess,'./checkpoint_2d_mean/model_'+str(i)+'.ckpt')
                    print ('Saved {}'.format(i))



train()

def test():
    with tf.Graph().as_default():
        learning_rate = 0.00001
        video,caption,caption_mask,output_logits,loss,dropout_prob = build_model()
        optim = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
        ckpt_file = './ckpt_v5/model_58000.ckpt.meta'
        saver = tf.train.Saver()
        with tf.Session() as sess:
            if ckpt_file:
                saver_ = tf.train.import_meta_graph(ckpt_file)
                saver_.restore(sess,'./ckpt_v5/model_58000.ckpt')
                print ("Restored model")
            else:
                sess.run(tf.initialize_all_variables())
            while(1):
                vid,caption_GT,_,current_batch_vids = fetch_data_batch_val(1)
                caps,caps_mask = convert_caption(['<BOS>'],word2id,30)

                for i in range(30):
                    o_l = sess.run(output_logits,feed_dict={video:vid,
                                                            caption:caps,
                                                            caption_mask:caps_mask,
                                                            dropout_prob:1.0})
                    out_logits = o_l.reshape([batch_size,n_steps-1,vocab_size])
                    output_captions = np.argmax(out_logits,2)
                    caps[0][i+1] = output_captions[0][i]
                    print_in_english(caps)
                    if id2word[output_captions[0][i]] == '<EOS>':
                        break
                print ('............................\nGT Caption:\n')
                print_in_english(caption_GT)
                play_video = input('Should I play the video? ')
                if play_video.lower() == 'y':
                    playVideo(current_batch_vids)
                test_again = input('Want another test run? ')
                if test_again.lower() == 'n':
                    break
test()

