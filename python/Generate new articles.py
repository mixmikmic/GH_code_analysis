import numpy as np 
import random  
import tensorflow as tf 
import datetime 

text = open('./data/wikitext-2-raw/wiki.test.raw').read()
print('text length in number of characters:', len(text))

print('head of text:')
print(text[:1000]) #all tokenized words, stored in a list called text

chars = sorted(list(set(text)))
char_size = len(chars)
print('number of characters:', char_size)
print(chars)

char2id = dict((c, i) for i, c in enumerate(chars))
id2char = dict((i, c) for i, c in enumerate(chars))
print(id2char)

def sample(prediction):
    #Samples are uniformly distributed over the half-open interval 
    r = random.uniform(0,1)
    #store prediction char
    s = 0
    #since length > indices starting at 0
    char_id = len(prediction) - 1
    #for each char prediction probabilty
    for i in range(len(prediction)):
        #assign it to S
        s += prediction[i]
        #check if probability greater than our randomly generated one
        if s >= r:
            #if it is, thats the likely next char
            char_id = i
            break
    #dont try to rank, just differentiate
    #initialize the vector
    char_one_hot = np.zeros(shape=[char_size])
    #that characters ID encoded
    #https://image.slidesharecdn.com/latin-150313140222-conversion-gate01/95/representation-learning-of-vectors-of-words-and-phrases-5-638.jpg?cb=1426255492
    char_one_hot[char_id] = 1.0
    return char_one_hot

#vectorize our data to feed it into model
len_per_section = 50
skip = 100
sections = []
next_chars = []

for i in range(0, len(text) - len_per_section, skip):
    sections.append(text[i: i + len_per_section])
    next_chars.append(text[i + len_per_section])

#matrix of section length by num of characters
X = np.zeros((len(sections), len_per_section, char_size))
#label column for all the character id's, still zero
y = np.zeros((len(sections), char_size))

#for each char in each section, convert each char to an ID
#for each section convert the labels to ids 
for i, section in enumerate(sections):
    for j, char in enumerate(section):
        X[i, j, char2id[char]] = 1
    y[i, char2id[next_chars[i]]] = 1
print(y)

X.shape, y.shape

batch_size = 100
#total iterations
max_steps = 72001
#how often to log?
log_every = 100
#how often to save?
save_every = 6000
#too few and underfitting
#Underfitting occurs when there are too few neurons 
#in the hidden layers to adequately detect the signals in a complicated data set.
#too many and overfitting
hidden_nodes = 1024
#starting text
test_start = 'I am thinking that'
#to save our model
checkpoint_directory = 'ckpt'

if tf.gfile.Exists(checkpoint_directory):
    tf.gfile.DeleteRecursively(checkpoint_directory)
tf.gfile.MakeDirs(checkpoint_directory)

print('training data size:', len(X))
print('approximate steps per epoch:', int(len(X)/batch_size))

# build the model
# computational graph
graph = tf.Graph()

with graph.as_default():
    # variables are placeholders
    # global_step = # of batches seen by the graph
    # everytime a batch is provided, the weights are updatd in the direction
    # that minimize the loss
    # global_step keeps track of # of batches seen so far
    
    global_step = tf.Variable(0)
    

