from natural_language_processing import * 
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# getting direct samples is easy in this case
print direct_sample()

# but if we only knew the conditional probabilities, y_given_x, and x_given_y, we need to use Gibbs sampling

def random_x_given_y(y):
    if y <= 7:
        return random.randrange(1, y)
    return random.randrange(y-6, 7)

def random_y_given_x(x):
    return x + roll_a_die()

def gibbs_sample(num_iters=100):
    x,y=1,2
    for x in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y

print gibbs_sample()

# Samples with Gibbs
gtotals, gxs = [], []
for x in range(1000):
    x,y = gibbs_sample()
    gxs.append(x)
    gtotals.append(y)
    
# Direct samples
totals, xs = [], []
for x in range(1000):
    x,y = direct_sample()
    xs.append(x)
    totals.append(y)
    

plt.title('Samples taken with gibbs sampler')
plt.plot(gxs, gtotals, 'b*')
plt.plot(xs, totals, 'r+')
plt.show()

documents = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

K = 4

# this creates the a weight
def topic_weight(d, word, k):
    """given a document and a word in that document,
    return the weight for the k-th topic"""

    return p_word_given_topic(word, k) * p_topic_given_document(k, d)

def sample_from(weights):
    total = sum(weights)
    rnd = total * random.random()       # uniform between 0 and total
    for i, w in enumerate(weights):
        rnd -= w                        # return the smallest i such that
        if rnd <= 0: return i           # sum(weights[:(i+1)]) >= rnd


def choose_new_topic(d, word):
    return sample_from([topic_weight(d, word, k)
                        for k in range(K)])

from gensim import corpora, models, similarities
from gensim.models import hdpmodel, ldamodel
from itertools import izip

dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(document) for document in documents]

lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=5)
corpus_lda = lda[corpus]

for i in range(0, lda.num_topics-1):
    print lda.print_topic(i)
    print



