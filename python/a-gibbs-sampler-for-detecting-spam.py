def read_data(path='data/spam-or-ham.txt'):
    fp = open(path, 'r')
    texts = []
    labels = []
    for line in fp:
        line = line.strip()
        if line:
            label, text = line.split('\t')
            labels.append(label)
            texts.append(text)
    fp.close()    
    return texts, labels
texts, labels = read_data()

for i in range(5): 
    print("Message %d [%s]:\n%s\n" % (i, labels[i], texts[i]))

from collections import Counter
def select_vocabulary(texts, V, max_cnt=10000):
    counter = Counter()
    for text in texts:
        for word in text.split():
            counter[word.lower()] += 1    
    words = [w for w in counter.keys() if counter[w] < max_cnt]
    words = sorted(words, key=lambda x: counter[x])
    return set(words[-V:])

V = 10000
vocabulary = select_vocabulary(texts, V)
word2id = {w:i for i, w in enumerate(vocabulary)}
id2word = {i:w  for i, w in enumerate(vocabulary)}

def build_corpus(texts, vocabulary):
    corpus = []
    for text in texts:
        words = [w.lower() for w in text.split() if w.lower() in vocabulary]
        ids = [word2id[w] for w in words]
        counter = Counter(ids)
        document = {(i,c) for i, c in counter.items()}
        corpus.append(document)
    return corpus

corpus = build_corpus(texts, vocabulary)

def sample_labels(J, gamma_pi):
    pi = beta(gamma_pi[0], gamma_pi[1])
    return binomial(1, pi, J)

def initialize(W, labels, gamma_pi, gamma_theta):
    N = len(W)
    M = len(labels)
    V = len(gamma_theta)

    L = sample_labels(N - M, gamma_pi)
    theta = dirichlet(gamma_theta, 2)

    C = np.zeros((2,))
    C += gamma_pi
    cnts = np.zeros((2, V))
    cnts += gamma_theta
    
    for d, l in zip(W, labels.tolist() + L.tolist()):
        for i, c in d: cnts[l][i] += c
        C[l] += 1

    return {'C':C, 'N':cnts, 'L':L, 'theta':theta}

def update(state, X):
    C = state['C']
    N = state['N']
    L = state['L']
    theta = state['theta']
    # Update the labels for all documents:
    for j, l in enumerate(L):
        # Drop document j from the corpus:
        for i, c in X[j]: N[l][i] -= c
        C[l] -= 1  
        # Compute the conditional probability that L[j] = 1:  
        if C[0] == 1: pi = 1.0
        elif C[1] == 1 <= 0: pi = 0.0 
        else:
            # compute the product of probabilities (sum of logs)
            d = np.sum(C) - 1
            v0 = np.log((C[0] - 1.0) / d)
            v1 = np.log((C[1] - 1.0) / d)
            for i, c in X[j]:
                v0 += c * np.log(theta[0,i])
                v1 += c * np.log(theta[1,i])
            m = max(v0, v1)
            v0 = np.exp(v0 - m)
            v1 = np.exp(v1 - m)
            pi = v1 / (v0 + v1)
        # Sample the new label from the conditional probability:
        l = binomial(1, pi)
        L[j] = l
        # Add document j back into the corpus:
        C[l] += 1
        for i, c in X[j]: N[l][i] += c
    # Update the topics:
    theta[0] = dirichlet(N[0])
    theta[1] = dirichlet(N[1])

def run_sampler(W, labels, iterations, gamma_pi, gamma_theta):
    state = initialize(W, labels, gamma_pi, gamma_theta)
    X = W[len(labels):]
    for t in range(iterations): update(state, X)
    return state['L']

def compute_accuracy(L_true, L_predicted):
    correct = 0
    for i, l in enumerate(L_predicted):
        if L_true[i] == l: correct += 1
    accuracy = float(correct)/len(L_predicted)
    return accuracy

def predict_spam_or_ham(N, p, iterations=100):
    gamma_pi = (1, 1)
    gamma_theta = [1] * V

    W = corpus[:N]
    n = int(N * p)
    labels_observed = np.array([0 if x == 'ham' else 1 for x in labels[:n]])
    labels_unobserved = np.array([0 if x == 'ham' else 1 for x in labels[n:N]])

    
    L = run_sampler(W, labels_observed, iterations, gamma_pi, gamma_theta)
    accuracy = compute_accuracy(labels_unobserved, L)
    return accuracy

get_ipython().magic('time accuracy = predict_spam_or_ham(N=10000, p=0.8, iterations=100)')
print(accuracy)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

N = 10000
#N = len(texts)
p = 0.8
n = int(N * p)
X_train = texts[:n]
Y_train = labels[:n]
X_test = texts[n:N]
Y_test = labels[n:N]


pipeline = Pipeline([
    ('vectorizer',  CountVectorizer(vocabulary=vocabulary)),
    ('classifier',  MultinomialNB()) 
])
get_ipython().magic('time pipeline.fit(X_train, Y_train)')
Y_predict = pipeline.predict(X_test)
accuracy = compute_accuracy(Y_test, Y_predict)
print(accuracy)



