class Token():
    """A simple class to represent the tokens."""
    def __init__(self, word, count, is_label):
        self.word = word
        self.count = count
        self.is_label = is_label
    def __str__(self):
        return '('+self.word+', '+str(self.count)+')'
    def __repr__(self):
        return '('+self.word+', '+str(self.count)+')'
    
def cmp_to_key(mycmp):
    """Convert a cmp= function into a key= function"""
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

get_ipython().run_cell_magic('bash', '', '\nmyshuf() {\n  perl -MList::Util=shuffle -e \'print shuffle(<>);\' "$@";\n}\n\nnormalize_text() {\n  tr \'[:upper:]\' \'[:lower:]\' | sed -e \'s/^/__label__/g\' | \\\n    sed -e "s/\'/ \' /g" -e \'s/"//g\' -e \'s/\\./ \\. /g\' -e \'s/<br \\/>/ /g\' \\\n        -e \'s/,/ , /g\' -e \'s/(/ ( /g\' -e \'s/)/ ) /g\' -e \'s/\\!/ \\! /g\' \\\n        -e \'s/\\?/ \\? /g\' -e \'s/\\;/ /g\' -e \'s/\\:/ /g\' | tr -s " " | myshuf\n}\n\nDATADIR=/media/mat/ssdBackupMat/Datasets/sentiment/yelp_review_full_csv\n\necho "Processing dataset ${DATADIR} ... " \ncat "${DATADIR}/train.csv" | normalize_text > "${DATADIR}/data.train"\ncat "${DATADIR}/test.csv" | normalize_text > "${DATADIR}/data.test"\necho "done." ')

import re

def comp(x, y):
    if x.is_label != y.is_label:
        return -1 if x.is_label < y.is_label else 1
    else: 
        return -1 if x.count > y.count else 1
    
class Dictionary():
    
    def __init__(self, MAX_VOCAB_SIZE=30000000, verbose=True, buckets=2000000, min_count=0):
        self.min_count = min_count
        self.MAX_VOCAB_SIZE = MAX_VOCAB_SIZE
        self.EOS = '</s>'#end of sentence
        self.word2int = [None] * self.MAX_VOCAB_SIZE #stores the position of each word in self.words
        self.words = [] #stores the surface forms of each word
        self.size = 0 #size of the vocabulary (labels + words)
        self.nwords = 0 #number of unique words (labels excluded)
        self.nlabels = 0 #number of unique labels
        self.ntokens = 0 #total number of tokens
        self.verbose = verbose
        self.bucket = buckets #number of buckets for the ngrams, the higher the lesser collisions we have
        
    def find(self, word):
        """Use open addressing to get the hashing value of a given word."""
        h = self.hash_word(word) % self.MAX_VOCAB_SIZE
        while self.word2int[h] is not None and self.words[self.word2int[h]].word != word: #open addressing
            h = (h + 1) % self.MAX_VOCAB_SIZE
        return h

    def hash_word(self, s):
        """Compute a simple hashing value based on the characters of the string."""
        h = 2166136261
        for i in range(len(s)):
            h = h ^ ord(s[i])
            h = h * 16777619
        return h
    
    def add(self, word):
        """Add a word to the vocabulary if necessary or just increment its counter."""
        h = self.find(word)
        self.ntokens += 1
        if self.word2int[h] is None:
            w_type = True if "__label__" in word else False
            t = Token(word, 1, w_type)
            self.words.append(t)
            self.word2int[h] = self.size
            self.size += 1
        else:
            self.words[self.word2int[h]].count += 1
            
    def read_from_file(self, file_stream):
        """Read a document, handle all the words in it, create the vocabulary"""
        min_threshold = self.min_count #words of frequency <= to min_threshold are ignored
        line = file_stream.readline()
        while line:
            line = re.split(' |\n|\t|\v|\f|\r|\0', line)
            line.append(self.EOS)
            for w in line:
                if w == '': continue
                self.add(w)
                if self.ntokens % 10000000 == 0 and self.verbose:
                    print("Read %d M words" % (self.ntokens/1000000))
                if self.size > 0.75 * self.MAX_VOCAB_SIZE:
                    min_threshold += 1
                    self.threshold(min_threshold)
            line = file_stream.readline()
        self.threshold(min_threshold)
        if self.verbose:
            print("\rRead %d M words" % (self.ntokens/1000000))
            print("Number of unique words: %d" % (self.nwords))
            print("Number of labels: %d" % (self.nlabels))
        if self.size == 0:
            print("Empty vocabulary. Try a smaller -minCount value.")
    
    def threshold(self, min_value):
        """Remove all the words with a count less that min_value."""
        self.words.sort(key=cmp_to_key(comp))
        self.words = [x for x in self.words if x.is_label or x.count > min_value]
        self.size = 0
        self.nwords = 0
        self.nlabels = 0
        self.word2int = [None] * self.MAX_VOCAB_SIZE
        for t in self.words:
            h = self.find(t.word)
            self.word2int[h] = self.size
            self.size += 1
            if not t.is_label: self.nwords += 1
            if t.is_label: self.nlabels += 1
                
    def add_ngrams(self, line_of_hs, n):
        """For a line of tokens, compute ann the n-grams, append them to the given list."""
        line_size = len(line_of_hs)
        for i in range(line_size):
            h = line_of_hs[i]
            for j in range(i+1, i+n):
                if j >= line_size: break
                h = h * 116049371 + line_of_hs[j]
                line_of_hs.append(self.nwords + (h % self.bucket))
                
    def get_id(self, word):
        """Return the id of the word in self.words list."""
        h = self.find(word)
        return self.word2int[h]
    
    def get_type(self, word_id):
        """Return if the word is a label or not."""
        return self.words[word_id].is_label

    def get_line(self, file_stream):
        """Return a list of word ids and a list of labels for one line."""
        n_tokens = 0
        words = []
        labels = []
        line = file_stream.readline()
        line = re.split(' |\n|\t|\v|\f|\r|\0', line)
        line.append(self.EOS)
        for w in line:
            if w == '': continue
            tid = self.get_id(w)
            if tid is None: 
                continue
            w_is_label = self.get_type(tid)
            n_tokens += 1
            if not w_is_label:
                words.append(tid)
            if w_is_label:
                labels.append(tid - self.nwords) #the self.words list is sorted with labels at the end
        return words, labels, n_tokens

data_train_path = '/media/mat/ssdBackupMat/Datasets/sentiment/yelp_review_full_csv/data.train'

min_count = 1
buckets = 10000000

dictionary = Dictionary(buckets=buckets, min_count=min_count)
file_stream = open(data_train_path, 'r')
dictionary.read_from_file(file_stream)

import pickle

with open('dictionary_301016_full.pickle', 'wb') as f:
    pickle.dump(dictionary, f)

import pickle

with open('dictionary_301016_full.pickle', 'rb') as f:
    dictionary = pickle.load(f)

def get_batch(d, file_stream, batch_size, ngrams=2):
    """Return a mini-batch of size BATCH_SIZE: batch_X and batch_Y 
    containing respectively the words ids and labels.
    
    Arguments:
        - d: dictionary to use
        - file_stream: where to read the data from
        - batch_size: the number of (document, label) pairs inside one batch
        - ngrams: the ngram parameter, 2 for bigrams
    
    Returns:
        - batch_X: a list of bag of tricks
        
                [[8,6,10,800,89846,123582,338745],
                 [5,2,6,89456,1654984],
                 [98,100,548,5,3,1,548998,102548,154789,132000,1459877]]
                 
        - batch_Y: a list of labels [1,4,0]
    
    """
    batch_X = []
    batch_Y = []
    tot_n_tokens = 0
    for i in range(batch_size):
        words, labels, n_tokens = d.get_line(file_stream)
        tot_n_tokens += n_tokens
        if len(labels) == 0: break
        if ngrams>1: d.add_ngrams(words, ngrams)
        batch_X.append(words)
        batch_Y.append(labels[0])
    return batch_X, batch_Y, tot_n_tokens

def batch_to_sparse(batch_X):
    """Take a mini-batch as input and returns the components sp_inputs_indices, 
    sp_inputs_ids_val, and sp_inputs_shape, necessary to create a sparse 
    tensorflow tensor."""
    sp_inputs_indices = []
    sp_inputs_ids_val = []
    max_size = 0
    for i in range(len(batch_X)):
        sp_inputs_indices += [[i, j] for j in range(len(batch_X[i]))] #e.g. [[0,0],[0,1],[0,2],[1,0],[2,0],[2,1]]
        max_size = max(max_size, len(batch_X[i]))
        sp_inputs_ids_val += batch_X[i]
    sp_inputs_shape = [len(batch_X), max_size]
    return sp_inputs_indices, sp_inputs_ids_val, sp_inputs_shape

def get_next_sparse_batch(d, file_stream, batch_size, ngrams=2):
    """Read from file_stream and generate sparse batches of size batch_size."""
    batch_X, batch_Y, n_tokens = get_batch(d, file_stream, batch_size, ngrams)
    in_indices, in_ids_val, in_shape = batch_to_sparse(batch_X)
    return in_indices, in_ids_val, in_shape, batch_Y, n_tokens

NUM_LABELS = dictionary.nlabels
INPUT_DIM = dictionary.nwords + dictionary.bucket
HIDDEN_SIZE = 10
NGRAMS = 2 #bigrams
BATCH_SIZE = 256
LEARNING_RATE = 1e-2
NUM_EPOCH = 1

tokens_processed = 0.
tokens_to_process = float(NUM_EPOCH * dictionary.ntokens)

print("NUM_LABELS: %d\nINPUT_DIM: %d\nTokens to process: %d" % (NUM_LABELS, INPUT_DIM, tokens_to_process))

import tensorflow as tf
import numpy as np
import threading

tf.reset_default_graph()#Reset the graph essential to use with jupyter else variable conflicts

class QueueCtrl(object):

    def __init__(self):
        """The init links the input tensors with the enqueue operation that will be used
        to fill the queue. 
        """
        self.sp_inputs_indices = tf.placeholder(tf.int64)
        self.sp_inputs_ids_val = tf.placeholder(tf.int64)
        self.sp_inputs_shape = tf.placeholder(tf.int64)
        self.inputs_num_tokens = tf.placeholder(tf.int64)
        
        self.labels = tf.placeholder(tf.int64, shape=(None), name='labels')
        
        self.queue = tf.FIFOQueue(dtypes=[tf.int64, tf.int64, tf.int64, tf.int64, tf.int64],
                                           capacity=500)

        self.enqueue_op = self.queue.enqueue([self.sp_inputs_indices,
                                              self.sp_inputs_ids_val,
                                              self.sp_inputs_shape,
                                              self.labels,
                                              self.inputs_num_tokens])

    def get_batch_from_queue(self):
        """Return one batch
        """
        return self.queue.dequeue()

    def thread_main(self, sess, coord, data_path):
        """Function nexecuted by the thread. Loop over the data, add the minibatches to the queue. 
        Stops when the coordinator says so.
        """
        print("Starting publisher thread: %d" % (threading.get_ident()))
        train_fs = open(data_path, 'r')
        while not coord.should_stop():
            in_indices, in_ids_val, in_shape, batch_Y, n_tokens = get_next_sparse_batch(dictionary, train_fs, 
                                                                                        BATCH_SIZE, NGRAMS)
            if not batch_Y: #EOF
                train_fs = open(data_path, 'r')
                in_indices, in_ids_val, in_shape, batch_Y, n_tokens = get_next_sparse_batch(dictionary, train_fs, 
                                                                                            BATCH_SIZE, NGRAMS)
            sess.run(self.enqueue_op, feed_dict={self.sp_inputs_indices:in_indices, 
                                                 self.sp_inputs_ids_val:in_ids_val,
                                                 self.sp_inputs_shape:in_shape,
                                                 self.labels:batch_Y,
                                                 self.inputs_num_tokens:n_tokens}) #append batch to the queue

    def start_thread(self, sess, coord, data_path):
        """Start the thread"""
        t = threading.Thread(target=self.thread_main, args=(sess, coord, data_path))
        t.daemon = True
        t.start()
        return [t]

queue_ctrl = QueueCtrl()
in_indices, in_ids_val, in_shape, batch_Y, n_tokens = queue_ctrl.get_batch_from_queue()

sp_inputs_ids = tf.SparseTensor(in_indices, in_ids_val, in_shape) 

embedding_matrix = tf.get_variable("embeddings", [INPUT_DIM, HIDDEN_SIZE], tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())

hidden_vectors = tf.nn.embedding_lookup_sparse(embedding_matrix, sp_inputs_ids, None, 
                                               name="averaged_embeddings",
                                               combiner="mean") #average the embeddings for each input

context_matrix = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, NUM_LABELS], 
                                                 mean=0.0, stddev=2./(NUM_LABELS+HIDDEN_SIZE), 
                                                 dtype=tf.float32, 
                                                 name='context_matrix'))

logits = tf.matmul(hidden_vectors, context_matrix)

lr = tf.Variable(LEARNING_RATE, trainable=False)

#Getting the probabilities and accuracy (Used at inference time):
probabilities = tf.nn.softmax(logits, name="softmax")
correct_predictions = tf.equal(tf.argmax(probabilities,1), batch_Y)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

#Defining the error metric used and loss (Used at training time):
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, batch_Y, name="cross_entropy")
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

#Initialization op:
init = tf.initialize_all_variables()

get_ipython().run_cell_magic('bash', '', 'nvidia-smi')

import time

def train_model(model_name):
    
    saver = tf.train.Saver()
    data_train_path = '/media/mat/ssdBackupMat/Datasets/sentiment/yelp_review_full_csv/data.train'
    start_time = time.time()
    PRINT_EVERY = 200

    sess = tf.Session()

    sess.run(init)
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)
    my_thread = queue_ctrl.start_thread(sess, coord, data_train_path) #start the publisher thread
    iteration, progress = 0, 0.
    tokens_processed = 0.
    
    print("\nTraining model: %s" % model_name)

    while progress < 100:
        _, loss_, n_tokens_ = sess.run([train_step, loss, n_tokens])
        tokens_processed += n_tokens_
        progress = tokens_processed / tokens_to_process *100.
        if iteration % PRINT_EVERY == 0:
            print("Iter: %d, %.2f%% done, Minibatch loss: %.4f, Elements in queue: %d" 
                  % (iteration, progress, loss_, sess.run(queue_ctrl.queue.size())))
        iteration += 1

    coord.request_stop()

    print("Done. Exec time: %.2f minutes." % ((time.time() - start_time) / 60.))
    save_path = saver.save(sess, model_name)

    coord.join(my_thread, stop_grace_period_secs=10)
    sess.close()

model_base_name = './model_cbow_yelp_f_291016'

model_names = []

for i in range(10):
    name = model_base_name + str(i) + '.ckpt'
    model_names.append(name)
    train_model(name)

data_test_path = '/media/mat/ssdBackupMat/Datasets/sentiment/yelp_review_full_csv/data.test'

def compute_accuracy(label_probas):
    """Takes as input a list of tuples [(true_label, [p1, p2, ...]), ...] it
    outputs the accuracy.
    """
    good_guesses = 0.
    total =0.
    for label, probas in label_probas:
        if label == np.argmax(probas):
            good_guesses += 1.
        total += 1.
    return good_guesses / total

accuracy = []

for model_path in model_names:
    
    predictions = []

    with tf.Session() as sess:

        saver.restore(sess, model_path)

        test_fs = open(data_test_path, 'r')
        in_indices_, in_ids_val_, in_shape_, batch_Y_, n_tokens = get_next_sparse_batch(dictionary, test_fs, 
                                                                                        512, NGRAMS)
        while batch_Y_:

            p = sess.run(probabilities, feed_dict={
              in_indices: in_indices_,  
              in_shape: in_shape_, 
              in_ids_val: in_ids_val_,
              batch_Y: batch_Y_
            })
            predictions += list(zip(batch_Y_, p))
            in_indices_, in_ids_val_, in_shape_, batch_Y_, n_tokens = get_next_sparse_batch(dictionary, test_fs, 
                                                                                            512, NGRAMS)
    accuracy.append(compute_accuracy(predictions) * 100.)

print("Accuracy of the model: %.2f%%, std: %.2f" % (np.mean(accuracy), np.std(accuracy)))

#TODO

class Token():
    """A simple class to represent the tokens."""
    def __init__(self, word, count, subwords):
        self.word = word
        self.count = count
        self.subwords = [] #will contain all the ngrams for this word
    def __str__(self):
        return '('+self.word+', '+str(self.count)+')'
    def __repr__(self):
        return '('+self.word+', '+str(self.count)+')'
    
def cmp_to_key(mycmp):
    """Convert a cmp= function into a key= function"""
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

import re
import random
import numpy as np

def comp(x, y):
    return -1 if x.count > y.count else 1
    
class Dictionary():
    
    def __init__(self, MAX_VOCAB_SIZE=30000000, verbose=True, buckets=2000000, min_count=0):
        self.min_count = min_count
        self.MAX_VOCAB_SIZE = MAX_VOCAB_SIZE
        self.MAX_LINE_SIZE = 1024
        self.EOS = '</s>'#end of sentence
        self.BOW = '<'#token added before the word to consider prefixes
        self.EOW = '>'#token added after the word to consider suffixes
        self.word2int = [None] * self.MAX_VOCAB_SIZE #stores the position of each word in self.words
        self.words = [] #stores the surface forms of each word
        self.size = 0 #size of the vocabulary (labels + words)
        self.nwords = 0 #number of unique words (labels excluded)
        self.ntokens = 0 #total number of tokens
        self.verbose = verbose
        self.pkeep = []
        self.sampling_threshold = 1e-4
        self.min_ngram_size = 3
        self.max_ngram_size = 6
        self.bucket = buckets #number of buckets for the ngrams, the higher the lesser collisions we have
        
    def find(self, word):
        """Use open addressing to get the hashing value of a given word."""
        h = self.hash_word(word) % self.MAX_VOCAB_SIZE
        while self.word2int[h] is not None and self.words[self.word2int[h]].word != word: #open addressing
            h = (h + 1) % self.MAX_VOCAB_SIZE
        return h

    def hash_word(self, s):
        """Compute a simple hashing value based on the characters of the string."""
        h = 2166136261
        for i in range(len(s)):
            h = h ^ ord(s[i])
            h = h * 16777619
        return h
    
    def init_ngrams(self):
        """Add the prefix and suffixe token to each word, add its idx to the list of subwords, 
        compute all the subwords"""
        for i in range(self.size):
            word = self.BOW + self.words[i].word + self.EOW
            self.words[i].subwords.append(i)
            self.compute_ngrams(word, i)
            
    def compute_ngrams(self, word, idx):
        """Compute all the sub-ngrams of size self.between min_ngram_size and 
        self.max_ngram_size, and give them an id in the range 
        [self.nwords, self.nwords+self.buckets]"""
        for i in range(len(word)):
            ngram = ""
            for j in range(i, i+self.max_ngram_size):
                if j >= len(word) or len(ngram) >= self.max_ngram_size:
                    break
                ngram += word[j]
                if self.min_ngram_size <= len(ngram):
                    h = self.hash_word(ngram) % self.bucket
                    self.words[idx].subwords.append(self.nwords + h)
    
    def add(self, word):
        """Add a word to the vocabulary if necessary or just increment its counter."""
        h = self.find(word)
        self.ntokens += 1
        if self.word2int[h] is None:
            t = Token(word, 1, [])
            self.words.append(t)
            self.word2int[h] = self.size
            self.size += 1
        else:
            self.words[self.word2int[h]].count += 1
            
    def read_from_file(self, file_stream):
        """Read a document, handle all the words in it, create the vocabulary"""
        min_threshold = self.min_count #words of frequency <= to min_threshold are ignored
        line = file_stream.readline()
        while line:
            line = re.split(' |\n|\t|\v|\f|\r|\0', line)
            for w in line:
                if w == '': continue
                self.add(w)
                if self.ntokens % 100000000 == 0 and self.verbose:
                    print("Read %d M words" % (self.ntokens/1000000))
                if self.size > 0.75 * self.MAX_VOCAB_SIZE:
                    min_threshold += 1
                    self.threshold(min_threshold)
            line = file_stream.readline()
        self.threshold(min_threshold)
        self.init_table_discard()
        self.init_ngrams()
        if self.verbose:
            print("\rRead %d M words" % (self.ntokens/1000000))
            print("Number of unique words: %d" % (self.nwords))
        if self.size == 0:
            print("Empty vocabulary. Try a smaller -minCount value.")
            
    def init_table_discard(self):
        """"""
        for i in range(self.size):
            f = float(self.words[i].count) / float(self.ntokens)
            self.pkeep.append(np.sqrt(self.sampling_threshold / f) + self.sampling_threshold / f)

    def discard(self, idx, rnd):
        """Keep the word with a probability of self.pkeep[idx]."""
        return rnd > self.pkeep[idx]
    
    def threshold(self, min_value):
        """Remove all the words with a count less that min_value."""
        self.words.sort(key=cmp_to_key(comp))
        self.words = [x for x in self.words if x.count > min_value]
        self.size = 0
        self.nwords = 0
        self.word2int = [None] * self.MAX_VOCAB_SIZE
        for t in self.words:
            h = self.find(t.word)
            self.word2int[h] = self.size
            self.size += 1
            self.nwords += 1
                
    def get_id(self, word):
        """Return the id of the word in self.words list."""
        h = self.find(word)
        return self.word2int[h]

    def get_line(self, file_stream):
        """Return a list of word ids and a list of labels for one line.
        Returns (-1,-1) if EOF."""
        n_tokens = 0
        words = []
        line = file_stream.readline()
        if line == '':
            return -1, -1
        line = re.split(' |\n|\t|\v|\f|\r|\0', line)
        for w in line:
            if w == '': continue
            tid = self.get_id(w)
            if tid is None: 
                continue
            n_tokens += 1
            if not self.discard(tid, random.random()): 
                words.append(tid)
            if len(words) > self.MAX_LINE_SIZE:
                break
        return words, n_tokens

fs = open('./wiki_corpus/wiki_full_word2vec.txt', 'r')

wiki_dict = Dictionary(min_count=5, buckets=2000000)
wiki_dict.read_from_file(fs)

import pickle

with open('dictionary_wiki_301016_full.pickle', 'wb') as f:
    pickle.dump(wiki_dict, f)

import pickle

with open('dictionary_wiki_301016_full.pickle', 'rb') as f:
    wiki_dict = pickle.load(f)

def get_pairs_from_line(line, d, window_size):
   """Use the window size to generate a batch of (ngrams, target_word) pairs."""
   batch_X = []
   batch_Y = []
   for i in range(len(line)):
       b = random.randint(1, window_size)
       ngrams = d.words[line[i]].subwords #the representation of the input
       for j in range(-b,b+1):
           if j != 0 and 0 <= i+j < len(line):
               batch_X.append(ngrams)
               batch_Y.append(line[i+j])
   return batch_X, batch_Y, len(batch_Y)

def generate_pairs_file(file_stream, d, window_size, out_stream):
   """Iterate over the entire corpus, generate all the pairs and save them in a file:
   
               [1, 25, 48, 7]|6 -> On the left: all the ids of the ngrams for the word in the middle of the window
               [156, 5, 9]|58   -> The label on the right is a word in the context
               ...
               
   """
   line, num_t = d.get_line(file_stream)
   while num_t != -1:
       batch_X, batch_Y, size = get_pairs_from_line(line, d, window_size)
       for i in range(size):
           ngrams = batch_X[i]
           target = batch_Y[i]
           out_stream.write(str(ngrams) + '|' + str(target) + '\n')
       line, num_t = d.get_line(file_stream)
       
def batch_to_sparse(batch_X):
   """Take a mini-batch as input and return the components sp_inputs_indices, sp_inputs_ids_val, and 
   sp_inputs_shape, necessary to create a sparse tensorflow tensor."""
   sp_inputs_indices = []
   sp_inputs_ids_val = []
   max_size = 0
   for i in range(len(batch_X)):
       sp_inputs_indices += [[i, j] for j in range(len(batch_X[i]))] #e.g. [[0,0],[0,1],[0,2],[1,0],[2,0],[2,1]]
       max_size = max(max_size, len(batch_X[i]))
       sp_inputs_ids_val += batch_X[i]
   sp_inputs_shape = [len(batch_X), max_size]
   return sp_inputs_indices, sp_inputs_ids_val, sp_inputs_shape

WINDOW_SIZE = 5

with open('./wiki_corpus/wiki_full_word2vec.txt', 'r') as in_stream:
    with open('/media/mat/ssdBackupMat/Datasets/Wikipedia/pairs_fasttext/pairs.txt', 'w') as out_stream:
        generate_pairs_file(in_stream, wiki_dict, WINDOW_SIZE, out_stream)

get_ipython().run_cell_magic('bash', '', 'wc -l /media/mat/ssdBackupMat/Datasets/Wikipedia/pairs_fasttext/wiki_60M_pairs.shuf')

def get_next_sparse_batch(file_stream, batch_size):
    """Read the file of pairs and generate a minibatch
    """
    batch_X = []
    batch_Y = []
    while len(batch_X) < batch_size:
        line = file_stream.readline()
        if line == '': #EOF
            break
        try:
            X, Y = line.split('|')
            X = eval(X)
            Y = eval(Y)
            batch_X.append(X)
            batch_Y.append([Y])
        except:
            continue
    in_indices, in_ids_val, in_shape = batch_to_sparse(batch_X)
    return in_indices, in_ids_val, in_shape, batch_Y

NUM_NEG_SAMPLES = 9
VOCAB_SIZE = wiki_dict.nwords
INPUT_SIZE = VOCAB_SIZE + wiki_dict.bucket
HIDDEN_SIZE = 80
BATCH_SIZE = 16
LEARNING_RATE = .02 
PORTION_TO_PROCESS = 1. #Will process 100% of the training set
NUM_EPOCH = 1
PAIRS_FILE_PATH = '/media/mat/ssdBackupMat/Datasets/Wikipedia/pairs_fasttext/wiki_30M_pairs.shuf'

pairs_processed = 0.
pairs_to_process = 30000000

print("VOCAB_SIZE: %d\nINPUT_SIZE: %d\nPairs to process: %d" % (VOCAB_SIZE, INPUT_SIZE, pairs_to_process))

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

sp_inputs_indices = tf.placeholder(tf.int64)
sp_inputs_ids_val = tf.placeholder(tf.int64)
sp_inputs_shape = tf.placeholder(tf.int64)

labels = tf.placeholder(tf.int64)

sp_inputs_ids = tf.SparseTensor(sp_inputs_indices, sp_inputs_ids_val, sp_inputs_shape) 

embedding_matrix = tf.get_variable("embeddings", [INPUT_SIZE, HIDDEN_SIZE], tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())

hidden_vectors = tf.nn.embedding_lookup_sparse(embedding_matrix, sp_inputs_ids, None, 
                                               name="sum_of_embeddings",
                                               combiner="sum") #sum the embeddings for each input

context_matrix = tf.get_variable("context_matrix", [VOCAB_SIZE, HIDDEN_SIZE], tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
out_biases = tf.Variable(tf.zeros([VOCAB_SIZE]), trainable=False)

loss = tf.reduce_mean(tf.nn.nce_loss(context_matrix, out_biases, hidden_vectors, labels,
                                     NUM_NEG_SAMPLES, VOCAB_SIZE))

lr = tf.placeholder(tf.float32, shape=[])
train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
init = tf.initialize_all_variables()

saver = tf.train.Saver()

get_ipython().run_cell_magic('bash', '', 'nvidia-smi')

import time

start_time = time.time()
PRINT_EVERY = 5000
DECAY_LR_EVERY = 5000 #every 50 iterations

lr_ = LEARNING_RATE

with tf.Session() as sess:

    sess.run(init)
    iteration, progress = 0, 0.
    
    train_fs = open(PAIRS_FILE_PATH, 'r')
    in_indices, in_ids_val, in_shape, batch_Y = get_next_sparse_batch(train_fs, BATCH_SIZE)

    while progress < PORTION_TO_PROCESS*100. and batch_Y: 

        _, loss_ = sess.run([train_step, loss], feed_dict={
                sp_inputs_indices:in_indices, 
                sp_inputs_ids_val:in_ids_val,
                sp_inputs_shape:in_shape,
                labels:batch_Y,
                lr: lr_
        }) 
        pairs_processed += float(len(batch_Y))
        progress = pairs_processed / pairs_to_process *100.
        if iteration % DECAY_LR_EVERY == 0:
            lr_ = max(1e-6, (1. - pairs_processed / (PORTION_TO_PROCESS * pairs_to_process))) * LEARNING_RATE
        if iteration % PRINT_EVERY == 0:
            print("Iter: %d, %.2f%% done, Minibatch loss: %.4f, lr: %.4f" % (iteration, progress, loss_, lr_))
        iteration += 1

        in_indices, in_ids_val, in_shape, batch_Y = get_next_sparse_batch(train_fs, BATCH_SIZE)

    print("Done. Exec time: %.2f hours." % ((time.time() - start_time) / 3600.))
    save_path = saver.save(sess, "./model_skipgram_wiki_291016.ckpt")

save_path = "./model_skipgram_wiki_291016.ckpt"

embedding_matrix = None

with tf.Session() as sess:
    
    saver.restore(sess, save_path)
    
    all_ngrams = [w.subwords for w in wiki_dict.words]
    
    in_indices_, in_ids_val_, in_shape_ = batch_to_sparse(all_ngrams)

    embedding_matrix = sess.run(hidden_vectors, feed_dict={
          sp_inputs_indices: in_indices_,  
          sp_inputs_shape: in_shape_, 
          sp_inputs_ids_val: in_ids_val_
    })

print("Shape of the embedding matrix:", embedding_matrix.shape)

random_embedding_matrix = None

with tf.Session() as sess:
    
    sess.run(init)
    
    all_ngrams = [w.subwords for w in wiki_dict.words]
    
    in_indices_, in_ids_val_, in_shape_ = batch_to_sparse(all_ngrams)

    random_embedding_matrix = sess.run(hidden_vectors, feed_dict={
          sp_inputs_indices: in_indices_,  
          sp_inputs_shape: in_shape_, 
          sp_inputs_ids_val: in_ids_val_
    })

print("Shape of the embedding matrix:", random_embedding_matrix.shape)

def normalize_array(mat):
    for idx in range(mat.shape[0]):
        mat[idx] /= np.linalg.norm(mat[idx])
        
def cosine_sim(mat, vec):
    """Assumes normalized arrays"""
    cosine_sims = np.inner(vec, mat)
    return np.argsort(cosine_sims)[::-1], cosine_sims

normalize_array(embedding_matrix)
normalize_array(random_embedding_matrix)

word2idx = {w: idx for idx, w in enumerate([w.word for w in wiki_dict.words])}

query_word = 'switzerland'
query_word_idx = word2idx[query_word]
query_word_embedding = embedding_matrix[query_word_idx]

print("Query word: %s" % query_word)
print("\nTop similarities with the trained model:")
top_sim, sims = cosine_sim(embedding_matrix, query_word_embedding)
for i in top_sim[:15]:
    print("%.2f -- %s" % (sims[i], wiki_dict.words[i].word))
    
print("\nOnly displaying words not containing the query word:")
i, inc = 0, 0
while i < 15:
    idx = top_sim[inc]
    word = wiki_dict.words[idx].word
    if query_word not in word:
        i += 1
        print("%.2f -- %s" % (sims[idx], wiki_dict.words[idx].word))
    inc += 1
    
print("\nTop similarities with the random initialized matrix:")
top_sim, sims = cosine_sim(random_embedding_matrix, query_word_embedding)
for i in top_sim[:10]:
    print("%.2f -- %s" % (sims[i], wiki_dict.words[i].word))

word2idx = {w: idx for idx, w in enumerate([w.word for w in wiki_dict.words])}

query_word = 'science'
query_word_idx = word2idx[query_word]
query_word_embedding = embedding_matrix[query_word_idx]

print("Query word: %s" % query_word)
print("\nTop similarities with the trained model:")
top_sim, sims = cosine_sim(embedding_matrix, query_word_embedding)
for i in top_sim[:15]:
    print("%.2f -- %s" % (sims[i], wiki_dict.words[i].word))
    
print("\nOnly displaying words not containing the query word:")
i, inc = 0, 0
while i < 15:
    idx = top_sim[inc]
    word = wiki_dict.words[idx].word
    if query_word not in word:
        i += 1
        print("%.2f -- %s" % (sims[idx], wiki_dict.words[idx].word))
    inc += 1

word2idx = {w: idx for idx, w in enumerate([w.word for w in wiki_dict.words])}

query_word = 'english-born'
query_word_idx = word2idx[query_word]
query_word_embedding = embedding_matrix[query_word_idx]

print("Query word: %s" % query_word)
print("\nTop similarities with the trained model:")
top_sim, sims = cosine_sim(embedding_matrix, query_word_embedding)
for i in top_sim[:25]:
    print("%.2f -- %s" % (sims[i], wiki_dict.words[i].word))

word2idx = {w: idx for idx, w in enumerate([w.word for w in wiki_dict.words])}

query_word = 'micromanaging'
query_word_idx = word2idx[query_word]
query_word_embedding = embedding_matrix[query_word_idx]

print("Query word: %s" % query_word)
print("\nTop similarities with the trained model:")
top_sim, sims = cosine_sim(embedding_matrix, query_word_embedding)
for i in top_sim[:25]:
    print("%.2f -- %s" % (sims[i], wiki_dict.words[i].word))

unigrams_embedding_matrix = None

with tf.Session() as sess:
    
    saver.restore(sess, save_path)
    
    all_ngrams = [[w.subwords[0]] for w in wiki_dict.words]
    
    in_indices_, in_ids_val_, in_shape_ = batch_to_sparse(all_ngrams)

    unigrams_embedding_matrix = sess.run(hidden_vectors, feed_dict={
          sp_inputs_indices: in_indices_,  
          sp_inputs_shape: in_shape_, 
          sp_inputs_ids_val: in_ids_val_
    })

print("Shape of the embedding matrix:", unigrams_embedding_matrix.shape)

normalize_array(unigrams_embedding_matrix)

word2idx = {w: idx for idx, w in enumerate([w.word for w in wiki_dict.words])}

query_word = 'science'
query_word_idx = word2idx[query_word]
query_word_embedding = unigrams_embedding_matrix[query_word_idx]

print("Query word: %s" % query_word)
print("\nTop similarities with the trained model:")
top_sim, sims = cosine_sim(unigrams_embedding_matrix, query_word_embedding)
for i in top_sim[:25]:
    print("%.2f -- %s" % (sims[i], wiki_dict.words[i].word))

get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")



