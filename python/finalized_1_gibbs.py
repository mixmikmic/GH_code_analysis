import numpy as np
import theano
from theano import tensor as T
theano_rng = T.shared_randomstreams.RandomStreams(1234)
W_values = np.array([[1,1],[1,1]], dtype=theano.config.floatX)
bvis_values = np.array([1,1], dtype=theano.config.floatX)
bhid_values = np.array([1,1], dtype=theano.config.floatX)
W = theano.shared(W_values)
vbias = theano.shared(bvis_values)
hbias = theano.shared(bhid_values)

def propup(vis, v0_doc_len):
        pre_sigmoid_activation = T.dot(vis, W) + T.dot(hbias.reshape([1,hbias.shape[0]]).T,v0_doc_len).T        #---------------------------[edited]
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

def sample_h_given_v(v0_sample, v0_doc_len):
    v0_doc_len = v0_doc_len.reshape([1,ipt.shape[0]])
    pre_sigmoid_h1, h1_mean = propup(v0_sample, v0_doc_len)
    h1_sample = theano_rng.binomial(size=h1_mean.shape,
                                         n=1, p=h1_mean,
                                         dtype=theano.config.floatX)
    return [pre_sigmoid_h1, h1_mean, h1_sample]

def propdown(hid):
    pre_softmax_activation = T.dot(hid, W.T) + vbias                               #---------------------------[edited]
    return [pre_softmax_activation, T.nnet.softmax(pre_softmax_activation)]

def sample_v_given_h(h0_sample, v0_doc_len):
    v0_doc_len = v0_doc_len.reshape([1,ipt.shape[0]])
    pre_softmax_v1, v1_mean = propdown(h0_sample)
    v1_sample = theano_rng.multinomial(size=None,
                                         n=v0_doc_len, pvals=v1_mean,
                                         dtype=theano.config.floatX)               #---------------------------[edited]
    v1_doc_len = v1_sample[0].sum(axis=1)
    return [pre_softmax_v1, v1_mean, v1_sample, v1_doc_len]

def gibbs_hvh(h0_sample, v0_doc_len):
    pre_softmax_v1, v1_mean, v1_sample, v1_doc_len = sample_v_given_h(h0_sample, v0_doc_len)
    pre_sigmoid_h1, h1_mean, h1_sample = sample_h_given_v(v1_sample, v0_doc_len)
    return [pre_sigmoid_h1[0], h1_mean[0], h1_sample[0],
            pre_softmax_v1, v1_mean, v1_sample[0], v1_doc_len]                        #---------------------------[edited]


ipt = T.matrix()
ipt_rSum = ipt.sum(axis=1)

pre_sigmoid_ph, ph_mean, ph_sample = sample_h_given_v(ipt, ipt_rSum)
chain_start = ph_sample

results, updates = theano.scan( fn = gibbs_hvh,
                                outputs_info = [None, None, chain_start, None, None, None, ipt_rSum],
                                n_steps=5)

hgv = theano.function( [ipt], outputs=results, updates = updates)

b = np.array([[1,6,],[1,3],[5,1]], dtype = theano.config.floatX)

output = hgv(b)
[out_1,out_2,out_3,out_4,out_5,out_6,out_7] = output
print(out_1)
print(out_2)
print(out_3)
print(out_4)
print(out_5)
print(out_6)
print(out_7)

import numpy as np
import theano
from theano import tensor as T
theano_rng = T.shared_randomstreams.RandomStreams(1234)
W_values = np.array([[.1,-.4],[5,.4]], dtype=theano.config.floatX)
bvis_values = np.array([0.5,-0.6], dtype=theano.config.floatX)
bhid_values = np.array([-2,1], dtype=theano.config.floatX)
W = theano.shared(W_values)
vbias = theano.shared(bvis_values)
hbias = theano.shared(bhid_values)

def propup(vis, v_doc_len):
        pre_sigmoid_activation = T.dot(vis, W) + T.dot(hbias.reshape([1,hbias.shape[0]]).T,v_doc_len).T        #---------------------------[edited]
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

def sample_h_given_v(v0_sample, v_doc_len):
    pre_sigmoid_h1, h1_mean = propup(v0_sample, v_doc_len)
    h1_sample = theano_rng.binomial(size=h1_mean.shape,
                                         n=1, p=h1_mean,
                                         dtype=theano.config.floatX)
    return [pre_sigmoid_h1, h1_mean, h1_sample]

def propdown(hid):
    pre_softmax_activation = T.dot(hid, W.T) + vbias                               #---------------------------[edited]
    return [pre_softmax_activation, T.nnet.softmax(pre_softmax_activation)]

def sample_v_given_h(h0_sample, v_doc_len):
    pre_softmax_v1, v1_mean = propdown(h0_sample)
    v1_sample = theano_rng.multinomial(size=None,
                                         n=v_doc_len, pvals=v1_mean,
                                         dtype=theano.config.floatX)               #---------------------------[edited]
    return [pre_softmax_v1, v1_mean, v1_sample]

def gibbs_hvh(h0_sample, v_doc_len):
    pre_softmax_v1, v1_mean, v1_sample = sample_v_given_h(h0_sample, v_doc_len)
    pre_sigmoid_h1, h1_mean, h1_sample = sample_h_given_v(v1_sample, v_doc_len)
    return [pre_softmax_v1,    v1_mean,    v1_sample[0],
            pre_sigmoid_h1[0], h1_mean[0], h1_sample[0] ]                        #---------------------------[edited]


ipt = T.matrix()
ipt_rSum = ipt.sum(axis=1).reshape([1,ipt.shape[0]])

pre_sigmoid_ph, ph_mean, ph_sample = sample_h_given_v(ipt, ipt_rSum)
chain_start = ph_sample

results, updates = theano.scan( fn = gibbs_hvh,
                                outputs_info = [None, None, None, None, None, chain_start],
                                non_sequences = ipt_rSum,
                                n_steps=2)

hgv = theano.function( [ipt], outputs=results, updates = updates)

b = np.array([[1,6,],[1,3],[5,1]], dtype = theano.config.floatX)
output = hgv(b)
[out_1,out_2,out_3,out_4,out_5,out_6] = output
print(out_1)
print(out_2)
print(out_3)
print(out_4)
print(out_5)
print(out_6)

# iter 0: use v0 to initialize [pre_sigmoid_ph, ph_mean, ph_sample]_0
bias = bvis_values.reshape([1,bvis_values.shape[0]])
doc_len = b[0,:].sum()
pre_sigmoid_ph0 = (T.dot(b[0,:],W_values) + T.dot(bias.T,doc_len).T).eval()
ph0_mean = T.nnet.sigmoid(pre_sigmoid_ph0).eval()
ph0_sample = theano_rng.binomial(size=ph0_mean.shape,n=1, p=ph0_mean, dtype=theano.config.floatX).eval()
print(pre_sigmoid_ph0)
print(ph0_mean)
print(ph0_sample)

# iter 1:
pre_softmax_v1 = T.dot(ph0_sample, W_values.T) + bias
v1_mean = T.nnet.softmax(pre_softmax_v1)
v1_sample = theano_rng.multinomial(size=None,n=doc_len, pvals=v1_mean, dtype=theano.config.floatX)
print(pre_softmax_v1.eval())
print(v1_mean.eval())
print(v1_sample.eval())

doc_len = theano.shared(b.sum(axis=1))
print(hbias.eval())
print(T.outer(hbias,doc_len).eval())
print(T.outer(doc_len,hbias).eval())
pre_sigmoid_ph0 = (T.dot(b,W) + T.outer(doc_len,hbias))
print(pre_sigmoid_ph0.eval())

doc_len = theano.shared(b.sum(axis=1))
v_star = T.zeros_like(theano.shared(b))
v_star = T.set_subtensor(v_star[:,0], doc_len )
print(v_star.eval())

import numpy as np
import theano
from theano import tensor as T
theano_rng = T.shared_randomstreams.RandomStreams(1234)
W_values = np.array([[1,1,1],[1,1,1]], dtype=theano.config.floatX).T #3 visibles and 2 hidden
bvis_values = np.array([1,1,1], dtype=theano.config.floatX)
bhid_values = np.array([0.5,0.5], dtype=theano.config.floatX)
#W_values = np.array([[.1,-.4],[5,.4],[-.5,.3]], dtype=theano.config.floatX)
#bvis_values = np.array([0.5,-0.6], dtype=theano.config.floatX)
#bhid_values = np.array([-2,1,2], dtype=theano.config.floatX)
W = theano.shared(W_values)
vbias = theano.shared(bvis_values)
hbias = theano.shared(bhid_values)

def propup(vis, v_doc_len):
        pre_sigmoid_activation = T.dot(vis, W) + T.outer(v_doc_len,hbias)        #---------------------------[edited]
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

def sample_h_given_v(v0_sample, v_doc_len):
    pre_sigmoid_h1, h1_mean = propup(v0_sample, v_doc_len)
    h1_sample = theano_rng.binomial(size=h1_mean.shape,
                                         n=1, p=h1_mean,
                                         dtype=theano.config.floatX)
    return [pre_sigmoid_h1, h1_mean, h1_sample]

def propdown(hid):
    pre_softmax_activation = T.dot(hid, W.T) + vbias                               #---------------------------[edited]
    return [pre_softmax_activation, T.nnet.softmax(pre_softmax_activation)]

def sample_v_given_h(h0_sample, v_doc_len):
    pre_softmax_v1, v1_mean = propdown(h0_sample)
    v1_sample = theano_rng.multinomial(size=None,
                                         n=v_doc_len, pvals=v1_mean,
                                         dtype=theano.config.floatX)               #---------------------------[edited]
    return [pre_softmax_v1, v1_mean, v1_sample]

def gibbs_hvh(h0_sample, v_doc_len):
    pre_softmax_v1, v1_mean, v1_sample = sample_v_given_h(h0_sample, v_doc_len)
    pre_sigmoid_h1, h1_mean, h1_sample = sample_h_given_v(v1_sample, v_doc_len)
    return [pre_softmax_v1,    v1_mean,    v1_sample,
            pre_sigmoid_h1, h1_mean, h1_sample ]                        #---------------------------[edited]


ipt = T.matrix()
ipt_rSum = ipt.sum(axis=1)

pre_sigmoid_ph, ph_mean, ph_sample = sample_h_given_v(ipt, ipt_rSum)
chain_start = ph_sample

results, updates = theano.scan( fn = gibbs_hvh,
                                outputs_info = [None, None, None, None, None, chain_start],
                                non_sequences = ipt_rSum,
                                n_steps=2 )

hgv = theano.function( [ipt], outputs=results, updates = updates)

b = theano.shared(np.array([[1,6,1],[1,3,2],[5,2,1],[5,1,2]], dtype = theano.config.floatX) )

b_sum = b.sum(axis=1) #.reshape([1,b.shape[0]])
print(hbias.eval())
print(b_sum.eval())
print(W_values)
print(T.dot(b,W).eval())
print(T.outer(b_sum,hbias).eval())
print((T.dot(b,W) + T.outer(b_sum,hbias)).eval())
print( T.nnet.sigmoid(T.dot(b,W) + T.outer(b_sum,hbias)).eval() )

[out1,out2,out3] = sample_h_given_v(b,b_sum)
print(out1.eval())
print(out2.eval())
print(out3.eval())

print(W.eval())
print(out1.eval())
print(vbias.eval())
print( (T.dot(out1,W.T)+ vbias).eval())
print('---------------------------------------------------')
[out11,out12,out13] = sample_v_given_h(out1, b_sum)
print(out11.eval())
print(out12.eval())
print(out13.eval())
print('---------------------------------------------------')
print(b.get_value())

out12.eval()

b = np.array( np.array([[1,6,1],[1,3,2],[5,2,1],[5,1,2]], dtype = theano.config.floatX) )
output = hgv(b)
[out_1,out_2,out_3,out_4,out_5,out_6] = output
print(out_1)
print(out_2)
print(out_3)
print(out_4)
print(out_5)
print(out_6)

def free_energy(v_sample, v_doc_len):
    wx_b = T.dot(v_sample, W) + T.outer(v_doc_len, hbias)                      #---------------------------[edited]
    vbias_term = T.dot(v_sample, vbias)
    hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
    return -hidden_term - vbias_term

T.mean(free_energy(b,b_sum)).eval()

a = theano_rng.multinomial(size=None, n=b.sum(axis=1), pvals=out_2[0], dtype=theano.config.floatX)
print(a.eval())

np_rng = np.random.RandomState(1234)
np.asarray(np_rng.uniform(
                    low=-4 * np.sqrt(6. / (5)),
                    high=4 * np.sqrt(6. / (5)),
                    size=(3, 4)
                ), dtype=theano.config.floatX)


#theano.sandbox.rng_mrg.MRG_RandomStreams(seed=12345).multinomial(size=None, n=b.sum(axis=1), pvals=out_2[0], dtype=theano.config.floatX)


T.shared_randomstreams.RandomStreams(1234).multinomial(size=None, n=b.sum(axis=1), pvals=out_2[0], dtype=theano.config.floatX)

