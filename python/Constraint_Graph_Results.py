import matplotlib
matplotlib.use('pdf')
get_ipython().magic('pylab inline')


import seaborn; seaborn.set_style('whitegrid')
import pandas, itertools, networkx, time
from pomegranate import *

NYSE = ["AAPL","XOM","MSFT","GOOGL","GOOG","BRK.A","JNJ","WFC","WMT","GE","PG","JPM","CVX","VZ","FB","KO","PFE","T","ORCL","BAC"]
FTSE = ["VED","BWNG","GFRD","TALK","HSBA","CLLN","BKG","LGEN","PSN","PHNX","SL","ADN","RDSB","BP","CWD","AMFW","DEB","RTN","RIO","COB"]
TSE = ["ASHAI","DENTSU","DOME","FUJITSU","GAS","KEISEI","MITSUI","NEG","NICHIREI","PANASONIC","SKY","SUMITOMO","TAIHEIYO","UNITIKA"]

market_data = None
for stock in NYSE:
    stock_data = pandas.read_csv('NYSE/{}.csv'.format(stock.lower()), sep=',', usecols=[1, 4]).values
    a = (stock_data[:-1,0] > stock_data[1:,1]).astype('int')[:239]
    b = (stock_data[:,1] > stock_data[:,0]).astype('int')[:239]
    if market_data is None:
        market_data = numpy.array([a, b])
    else:
        market_data = numpy.vstack([market_data, a, b])
    
for stock in FTSE:
    stock_data = pandas.read_csv('FTSE/{}.csv'.format(stock.lower()), sep=',', usecols=[1, 4]).values
    a = (stock_data[:-1,0] > stock_data[1:,1]).astype('int')[:239]
    b = (stock_data[:,1] > stock_data[:,0]).astype('int')[:239]
    market_data = numpy.vstack([market_data, a, b])
    
for stock in TSE:
    try:
        stock_data = pandas.read_csv('TSE/{}.csv'.format(stock), sep=',', usecols=[1, 2]).values
    except:
        stock_data = pandas.read_csv('TSE/{}.csv'.format(stock), sep=',', usecols=[1, 2], encoding='utf-16').values
    a = (stock_data[:-1,0] > stock_data[1:,1]).astype('int')[:239]
    b = (stock_data[:,1] > stock_data[:,0]).astype('int')[:239]
    market_data = numpy.vstack([market_data, a, b])
    
market_data = market_data.T

nyse_open = tuple(range(0, 40, 2))
nyse_close = tuple(range(1, 40, 2))
ftse_open = tuple(range(40, 80, 2))
ftse_close = tuple(range(41, 80, 2))
tse_open = tuple(range(80, 108, 2))
tse_close = tuple(range(81, 108, 2))


names = []
for name in NYSE + FTSE + TSE:
    names.append(name + "-open")
    names.append(name + "-close")

cg = networkx.DiGraph()
cg.add_edge(tse_open, tse_close)
cg.add_edge(tse_open, ftse_open)
cg.add_edge(tse_close, ftse_close)
cg.add_edge(ftse_open, ftse_close)
cg.add_edge(ftse_open, nyse_open)
cg.add_edge(ftse_close, nyse_close)
cg.add_edge(nyse_open, nyse_close)
cg.add_edge(nyse_open, ftse_close)

get_ipython().magic("timeit BayesianNetwork.from_samples(market_data, algorithm='exact', constraint_graph=cg, state_names=names)")
get_ipython().magic("timeit BayesianNetwork.from_samples(market_data, algorithm='exact', constraint_graph=cg, state_names=names, n_jobs=4)")
get_ipython().magic("timeit BayesianNetwork.from_samples(market_data, algorithm='exact', constraint_graph=cg, state_names=names, max_parents=3)")

times = []
for i in range(1, 9):
    tic = time.time()
    BayesianNetwork.from_samples(market_data, algorithm='exact', constraint_graph=cg, state_names=names, n_jobs=i)
    times.append(time.time() - tic)

plt.title("Time to Learn Network", fontsize=14)
plt.plot(range(1, 9), times, c='c')
plt.xlabel("Number of Cores", fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel("Time (s)", fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(1, 8)

from sklearn.datasets import load_digits
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

X, y = load_digits(10, True)

X_train = X[:1500]
y_train = y[:1500]

X_test = X[1500:]
y_test = y[1500:]

mu = X_train.mean()
X_train = X_train > mu
X_test = X_test > mu

print "Multinomial naive Bayes (sklearn)"
tic = time.time()
clf = MultinomialNB().fit(X_train, y_train)
t = time.time() - tic
print "Accuracy: {:4.4}".format(clf.score(X_test, y_test))
print "Time: {:4.4}".format(t)
print "\n"

print "Random forest (sklearn)"
tic = time.time()
clf = RandomForestClassifier(100).fit(X_train, y_train)
t = time.time() - tic
print "Accuracy: {:4.4}".format(clf.score(X_test, y_test))
print "Time: {:4.4}".format(t)
print "\n"

cg = networkx.DiGraph()
cg.add_edge((64,), tuple(range(64)))

X_train = numpy.hstack((X_train, y_train.reshape(1500, 1)))
X_test = numpy.hstack((X_test, y_test.reshape(297, 1)))

tic = time.time()
model = BayesianNetwork.from_samples(X_train, algorithm='exact', constraint_graph=cg)
t = time.time() - tic

y_pred = numpy.zeros(297)
for i in range(297):
    y_pred[i] = model.predict_proba({model.states[j].name: X_test[i, j] for j in range(64)})[64].mle()

print "Bayesian network classifier (pomegranate)"
print "Accuracy: {:4.4}".format((y_pred == y_test).mean())
print "Time: {:4.4}".format(t)
print "\n"

