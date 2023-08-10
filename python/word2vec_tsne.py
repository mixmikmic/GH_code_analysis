import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Flatten, Activation, Merge
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams, make_sampling_table

from glob import glob
text_files = glob('./data/sotu/*.txt')

def text_generator():
    for path in text_files:
        with open(path, 'r') as f:
            yield f.read()
            
len(text_files)
print(text_files[0:5])

# our corpus is small enough where we
# don't need to worry about this, but good practice
max_vocab_size = 50000

# `filters` specify what characters to get rid of
tokenizer = Tokenizer(num_words=max_vocab_size,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n\'`“”–')

# fit the tokenizer
tokenizer.fit_on_texts(text_generator())


# we also want to keep track of the actual vocab size
# we'll need this later
# note: we add one because `0` is a reserved index in keras' tokenizer
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

embedding_dim = 256

pivot_model = Sequential()
pivot_model.add(Embedding(vocab_size, embedding_dim, input_length=1))

context_model = Sequential()
context_model.add(Embedding(vocab_size, embedding_dim, input_length=1))

# merge the pivot and context models
model = Sequential()
model.add(Merge([pivot_model, context_model], mode='dot', dot_axes=2))
model.add(Flatten())

# the task as we've framed it here is
# just binary classification,
# so we want the output to be in [0,1],
# and we can use binary crossentropy as our loss
model.add(Activation('sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')

pivot_model.summary()

context_model.summary()

model.summary()

n_epochs = 60

# used to sample words (indices)
#Generates a word rank-based probabilistic sampling table.
sampling_table = make_sampling_table(vocab_size)
print(len(sampling_table))
print(sampling_table)


for i in range(n_epochs):
    loss = 0
    for seq in tokenizer.texts_to_sequences_generator(text_generator()):
        # generate skip-gram training examples
        # - `couples` consists of the pivots (i.e. target words) and surrounding contexts
        # - `labels` represent if the context is true or not
        # - `window_size` determines how far to look between words
        # - `negative_samples` specifies the ratio of negative couples
        #    (i.e. couples where the context is false)
        #    to generate with respect to the positive couples;
        #    i.e. `negative_samples=4` means "generate 4 times as many negative samples"
        couples, labels = skipgrams(seq, vocab_size, window_size=5, negative_samples=4, sampling_table=sampling_table)
        if couples:
            pivot, context = zip(*couples)
            pivot = np.array(pivot, dtype='int32')
            context = np.array(context, dtype='int32')
            labels = np.array(labels, dtype='int32')
            loss += model.train_on_batch([pivot, context], labels)
    print('epoch %d, %0.02f'%(i, loss))

embeddings = model.get_weights()[0]

embeddings=embeddings[0]

embeddings.shape

word_index = tokenizer.word_index
reverse_word_index = {v: k for k, v in word_index.items()}

def get_embedding(word):
    idx = word_index[word]
    # make it 2d
    return embeddings[idx][:,np.newaxis].T

from scipy.spatial.distance import cdist

ignore_n_most_common = 50

def get_closest(word):
    embedding = get_embedding(word)

    # get the distance from the embedding
    # to every other embedding
    distances = cdist(embedding, embeddings)[0]

    # pair each embedding index and its distance
    distances = list(enumerate(distances))

    # sort from closest to furthest
    distances = sorted(distances, key=lambda d: d[1])

    # skip the first one; it's the target word
    for idx, dist in distances[1:]:
        # ignore the n most common words;
        # they can get in the way.
        # because the tokenizer organized indices
        # from most common to least, we can just do this
        if idx > ignore_n_most_common:
            return reverse_word_index[idx]

word='freedom'
idx = word_index[word]
print(idx)

print(get_closest('freedom'))
print(get_closest('justice'))
print(get_closest('america'))
print(get_closest('citizens'))
print(get_closest('citizen'))

with open('./results/embeddings.dat', 'w') as f:
    f.write('{} {}'.format(vocab_size, embedding_dim))

    for word, idx in word_index.items():
        embedding = ' '.join(str(d) for d in embeddings[idx])
        f.write('\n{} {}'.format(word, embedding))

#from gensim.models.doc2vec import Word2VecKeyedVectors
#w2v = Word2VecKeyedVectors.load_word2vec_format('embeddings.dat', binary=False)
#print(w2v.most_similar(positive=['freedom']))

#!pip install sklearn

from sklearn.manifold import TSNE

# `n_components` is the number of dimensions to reduce to
tsne = TSNE(n_components=2, verbose=1)

embeddings.shape

# apply the dimensionality reduction
# to our embeddings to get our 2d points
points = tsne.fit_transform(embeddings[0:2000,:])

print(points)

points.shape

#!pip install matplotlib

#import matplotlib
#matplotlib.use('Agg') # for pngs
import matplotlib.pyplot as plt

# plot our results
# make it quite big so we can see everything
fig, ax = plt.subplots(figsize=(40, 20))

# extract x and y values separately
xs = points[:,0]
ys = points[:,1]

# plot the points
# we don't actually care about the point markers,
# just want to automatically set the bounds of the plot
ax.scatter(xs, ys, alpha=0)
#for i, point in enumerate(points):
for i in range(1,500): 
    actstr=reverse_word_index.get(i)
    if (actstr.isalpha()==True):
               ax.annotate(actstr,(xs[i], ys[i]),fontsize=18)
    else:
        pass
plt.show()



