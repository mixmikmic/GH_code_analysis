import pandas as pd
import numpy as np
from datetime import *

features = pd.read_pickle('data/all_features.pkl')
posts = pd.read_pickle('data/all_posts.pkl')

features.head()

posts.head()

comp_site = {'n_ys':{},'n_vc':{},'n_mn':{},'n_nbw':{}}

def comp_site_map(row):
    for c in row['cname']:
        c = ' '.join(c)
        cur_count = comp_site['n_'+row['site']].get(c,0)
        comp_site['n_'+row['site']][c] = cur_count+1
        
posts.apply(comp_site_map,axis=1)

comp_site

features = pd.concat([features,pd.DataFrame(comp_site)],axis=1)

def getavg(fundamts):
    amts = map(float,str(fundamts).split('-'))
    return sum(amts)/len(amts)

features['fund_avg'] = features['fundamts'].apply(getavg)

hqs = list(features['hq'].unique())

features['founded'] = pd.to_datetime(features['founded'])

#junotele,name,netmeds,trideal,

features = features.T

del features['junotele']
del features['name']
del features['netmeds']
del features['trideal']
del features['bigbasket.com']
del features['croak.it']

features = features.T

features['months'] = features['founded'].apply(lambda x:(datetime.now()-x).days/30)

del features['founded']
del features['fundamts']

features = features.fillna(value=0)

features

for hq in hqs:
    features[hq] = (features['hq']==hq).astype('int64')

features

del features['name']

del features['hq']

features.ix['eventifier','fundamt'] = 2.5
features.ix['gapoon','fundamt'] = 0.17
features.ix['gapoon','fund_avg'] = 0.17

features_un = features.copy()

features = features_un.copy()

#normalize

for col in features.columns[201:]:
    features[col] = features[col].astype('float32')
    m = features[col].mean()
    sd = features[col].std()
    features[col] = features[col].apply(lambda x: (x-m)/sd)

#no normalize
for col in features.columns[201:]:
    features[col] = features[col].astype('float32')

del features[np.nan]

#ad-hoc for now
features['set'] = 'train'
features.ix['freecharge','set'] = 'test'
features.ix['frilp','set'] = 'test'
features.ix['burrp','set'] = 'test'
features.ix['magicbricks','set'] = 'test'

X = features[features['set']=='train']

y = X['acquired'].astype('float32')
del X['acquired']

del X['set']

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(penalty='l2',max_iter=100,solver='liblinear',n_jobs=-1)

logreg.fit(X,y)

X_test = features[features['set']=='test']
y_test = X_test['acquired']
del X_test['acquired']
del X_test['set']

X_test

y_test_pred = logreg.predict(X_test)

y_test_pred

y_test

features.ix[:,201:]  = features.ix[:,201:]*10

import random
import theano
import theano.tensor as T


companies = list(features.index)
len(companies)

comp_pred = pd.DataFrame({'acquired': features['acquired'],
                          'p_acquire':0,'p_unacquire':0,
                          'count':0},index=features.index)

features.ix[:,200:]

#ad-hoc for now
for epoch in range(len(companies)):
    features['set'] = 'train'
    
    features.ix[companies[epoch],'set'] = 'test'

    X = features[features['set']=='train']
    y = X['acquired'].astype('float32')
    del X['acquired']
    del X['set']
    
    X_test = features[features['set']=='test']
    X_index = X_test.index
    y_test = X_test['acquired']
    del X_test['acquired']
    del X_test['set']
    
    X_train = X.copy()
    y_train = y.copy()
    
    X_train = np.array(X_train.astype('float32'))
    X_test = np.array(X_test.astype('float32'))
    
    y_unac = (y_train==0).astype('float32')
    y_ac = (y_train==1).astype('float32')
    y_onehot = pd.DataFrame({'unac':y_unac,'yac':y_ac})
    y_onehot = np.array(y_onehot.astype('float32'))
    
    y_train = np.array(y_train).astype('float32')
    
    X = theano.shared(X_train.astype('float32')) # initialized on the GPU
    y = theano.shared(y_onehot.astype('float32'))
    test = T.matrix('test')

    num_examples = len(X_train)
    nn_input_dim = 229 # input layer dimensionality
    nn_output_dim = 2 # output layer dimensionality
    nn_hdim2 = 250
    nn_hdim1 = 150
    nn_hdim = 50
    # Gradient descent parameters
    epsilon = np.float32(0.01) # learning rate for gradient descent
    reg_lambda = np.float32(0.01) # regularization strength 

    # Shared variables with initial values. We need to learn these.
    W1 = theano.shared(np.random.randn(nn_input_dim, nn_hdim2).astype('float32'), name='W1')
    b1 = theano.shared(np.zeros(nn_hdim2).astype('float32'), name='b1')
    W2 = theano.shared(np.random.randn(nn_hdim2, nn_hdim1).astype('float32'), name='W2')
    b2 = theano.shared(np.zeros(nn_hdim1).astype('float32'), name='b2')
    W3 = theano.shared(np.random.randn(nn_hdim1, nn_hdim).astype('float32'), name='W2')
    b3 = theano.shared(np.zeros(nn_hdim).astype('float32'), name='b2')
    W4 = theano.shared(np.random.randn(nn_hdim, nn_output_dim).astype('float32'), name='W2')
    b4 = theano.shared(np.zeros(nn_output_dim).astype('float32'), name='b2')

    # Forward propagation
    z1 = X.dot(W1) + b1
    a1 = T.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = T.tanh(z2)
    z3 = a2.dot(W3) + b3
    a3 = T.tanh(z3)
    z4 = a3.dot(W4) + b4
    y_hat = T.nnet.softmax(z4) # output probabilties

    z1t = test.dot(W1) + b1
    a1t = T.tanh(z1t)
    z2t = a1t.dot(W2) + b2
    a2t = T.tanh(z2t)
    z3t = a2t.dot(W3) + b3
    a3t = T.tanh(z3t)
    z4t = a3t.dot(W4) + b4
    y_hatt = T.nnet.softmax(z4t)
    prediction_c = y_hatt

    # The regularization term
    loss_reg = 1./num_examples * reg_lambda/2 * (T.sum(T.sqr(W1))
                                                 + T.sum(T.sqr(W2)) 
                                                 + T.sum(T.sqr(W3))
                                                + T.sum(T.sqr(W4))) 
    # the loss function we want to optimize
    loss = T.nnet.categorical_crossentropy(y_hat, y).mean() + loss_reg

    # Returns a class prediction
    prediction = T.argmax(y_hat, axis=1)

    # Theano functions that can be called from our Python code
    forward_prop = theano.function([], y_hat)
    calculate_loss = theano.function([], loss)
    predict = theano.function([], prediction)
    predict_one = theano.function([test],prediction_c)

    dW4 = T.grad(loss, W4)
    db4 = T.grad(loss, b4)
    dW3 = T.grad(loss, W3)
    db3 = T.grad(loss, b3)
    dW2 = T.grad(loss, W2)
    db2 = T.grad(loss, b2)
    dW1 = T.grad(loss, W1)
    db1 = T.grad(loss, b1)

    gradient_step = theano.function(
        [],
        updates=((W4, W4 - epsilon * dW4),
                 (W3, W3 - epsilon * dW3),
                 (W2, W2 - epsilon * dW2),
                 (W1, W1 - epsilon * dW1),
                 (b4, b4 - epsilon * db4),
                 (b3, b3 - epsilon * db3),
                 (b2, b2 - epsilon * db2),
                 (b1, b1 - epsilon * db1)))

    def build_model(num_passes=501, print_loss=False):

        # Re-Initialize the parameters to random values. We need to learn these.
        # (Needed in case we call this function multiple times)
        np.random.seed(0)
        W1.set_value((np.random.randn(nn_input_dim, nn_hdim2) / np.sqrt(nn_input_dim)).astype('float32'))
        b1.set_value(np.zeros(nn_hdim2).astype('float32'))
        W2.set_value((np.random.randn(nn_hdim2, nn_hdim1) / np.sqrt(nn_hdim2)).astype('float32'))
        b2.set_value(np.zeros(nn_hdim1).astype('float32'))
        W3.set_value((np.random.randn(nn_hdim1, nn_hdim) / np.sqrt(nn_hdim1)).astype('float32'))
        b3.set_value(np.zeros(nn_hdim).astype('float32'))
        W4.set_value((np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)).astype('float32'))
        b4.set_value(np.zeros(nn_output_dim).astype('float32'))

        # Gradient descent. For each batch...
        for i in xrange(0, num_passes):
            # This will update our parameters W2, b2, W1 and b1!
            gradient_step()

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 200 == 0:
                print "Loss after iteration %i: %f" %(i, calculate_loss())

    print epoch,companies[epoch]
    build_model(print_loss=True)
    
    y_test_pred = pd.DataFrame(predict_one(X_test),index=X_index)
    y_test_pred = y_test_pred*100
    
    #y_test_pred.reset_index(inplace=True)
    y_test_pred['unacquire'] = y_test_pred[0]
    y_test_pred['acquire'] = y_test_pred[1]
    del y_test_pred[0]
    del y_test_pred[1]
    
    def add_to_preds(row):
        comp_pred.ix[row['index'],'count'] = comp_pred.ix[row['index'],'count']+1
        comp_pred.ix[row['index'],'p_acquire'] = comp_pred.ix[row['index'],'p_acquire']+row['acquire']
        comp_pred.ix[row['index'],'p_unacquire'] = comp_pred.ix[row['index'],'p_unacquire']+row['unacquire']

    y_test_pred.reset_index(inplace=True)
    y_test_pred.apply(add_to_preds,axis=1)

y_test_pred

y_test

#13
comp_pred[(comp_pred['acquired']=='1') & (comp_pred['p_acquire']>comp_pred['p_unacquire'])]

comp_pred['acc_percent'] = comp_pred['p_acquire']/comp_pred['count']

comp_pred.sort_values('acc_percent',ascending=False,inplace=True)

comp_pred['colors'] = comp_pred['acquired'].apply(lambda x: '#00FF00' if x=='0' else '#FF0000')

comp_pred

comp_pred.reset_index(inplace=True)
comp_pred['index2'] = comp_pred['index']
comp_pred.set_index('index2',inplace=True)

#del comp_pred['level_0']

from bokeh.charts import Bar, output_file, show
from bokeh.charts.attributes import ColorAttr, CatAttr
import bokeh.plotting as bk
bk.output_notebook()

p = Bar(comp_pred[:50]
        ,CatAttr(columns=['index'], sort=False)
        ,values='acc_percent',color='colors',ylabel='% chance to acquire',xlabel='company')
show(p)



