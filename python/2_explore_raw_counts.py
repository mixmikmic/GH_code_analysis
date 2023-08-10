import numpy as np
import heapq
import matplotlib.pyplot as plt


# import local code files
import sys, os
sys.path.append(os.getcwd() + '/code/')

from save import load_vocabulary, load_matrix
from explore_counts_fun import top_counts_bar_plot, co_counts_intersection

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

w2i, i2w = load_vocabulary('data/vocab_small_ex.txt')
co_counts = load_matrix('data/co_counts_small_ex')
word_counts = np.load('data/word_counts_small_ex.npy')

# uncomment this code if you have the larger data file
# w2i, i2w = load_vocabulary('data_no_github/vocab_10000.txt')
# co_counts = load_matrix('data_no_github/co_counts_10000')
# word_counts = np.load('data_no_github/word_counts_10000.npy')

co_counts

print(w2i['lawyer'])
print(i2w[15788])

print(co_counts[w2i['lawyers'], w2i['criminal']])

word_counts[w2i['lawyer']]

N = 20
heapq.nlargest(N, zip(word_counts, i2w)) # this piece of code finds the largest values of total_word_counts

top_counts_bar_plot(word_counts,
                    i2w,
                    N=20,
                    title='',
                    figsize=[10, 10])

word = 'violent'

# vector of co-occurence counts for word
# the .toarray().reshape(-1) converts the row vector to a numpy array
word_co_counts = co_counts[w2i[word], :].toarray().reshape(-1) 

top_counts_bar_plot(word_co_counts,
                    i2w,
                    N=50,
                    title='top words co-occuring with %s' % word,
                    figsize=[10, 10])

# plt.figure(figsize=[10, 10])
# plt.hist(word_coo_counts, bins=1000)#np.arange(max(word_coo_counts)));
# plt.xlim([0, max(word_coo_counts)])
# plt.xlabel('counts')
# plt.title('histogram of co-occurence couts for all words with %s'% word)


# print 'mean: %f' % np.mean(word_coo_counts)
# print 'var: %f' % np.var(word_coo_counts)

def similarity(word1, word2, sim='angle'):
    """
    Computes the similarity between two words
    
    Parameters
    ----------
    word1, word2: words to compare
    sim: which similarity measure to use (angle, cosine, jaccard, dice)
    
    Returns
    -------
    similarity measure between two words
    """
    
    v1 = vec(word1)
    v2 = vec(word2)

    if sim == 'angle':
        return angle_between(v1, v2)
    elif sim == 'cosine':
        return cosine_sim(v1, v2)
    elif sim == 'jaccard':
        return jaccard_sim(v1, v2)
    elif sim == 'dice':
        return dice_sim(v1, v2)
    else:
        raise ValueError('sim must be one of: angle, cosine, jaccard, dice')

def vec(word):
    """
    Returns the vector for word as an array
    """
    return co_counts[w2i[word], :].toarray().reshape(-1)

def cosine_sim(v, w):
    return np.dot(v, w) / np.sqrt(np.dot(v, v) * np.dot(w, w))

def angle_between(v, w):
    cos_angle = cosine_sim(v, w)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def jaccard_sim(v, w):
    return np.minimum(v, w).sum()/np.maximum(v, w).sum()

def dice_sim(v, w):
     return 2.0 * np.minimum(v, w).sum() /(v + w).sum()

word1 = 'lawyer'
word2 = 'criminal'

print('similarity between %s and %s' % (word1, word2))
print()
print('angle: %f' % similarity(word1, word2, sim='angle'))
print('cosine: %f' % similarity(word1, word2, sim='cosine'))
print('jaccard: %f' % similarity(word1, word2, sim='jaccard'))
print('dice: %f' % similarity(word1, word2, sim='dice'))

word1 = 'lawyer'
word2 = 'lawyers'
words_both, c1, c2 = co_counts_intersection(co_counts, word1, word2, w2i, i2w, threshold=20)
words_both



