get_ipython().magic('matplotlib inline')
import utils
from utils import *
from __future__ import division, print_function

path = './dataset/glove.6B/'
#res_path = path+'/glove.6B/'
res_path = path

def get_glove(name):
    with open(path+ 'glove.' + name + '.txt', 'r', encoding='utf8') as f: lines = [line.split() for line in f]
    words = [d[0] for d in lines]
    vecs = np.stack(np.array(d[1:], dtype=np.float32) for d in lines)
    wordidx = {o:i for i,o in enumerate(words)}
    save_array(res_path+name+'.dat', vecs)
    pickle.dump(words, open(res_path+name+'_words.pkl','wb'))
    pickle.dump(wordidx, open(res_path+name+'_idx.pkl','wb'))

#get_glove('6B.50d')
get_glove('6B.100d')
get_glove('6B.200d')
get_glove('6B.300d')

def load_glove(loc):
    return (load_array(loc+'.dat'),
        pickle.load(open(loc+'_words.pkl','rb')),
        pickle.load(open(loc+'_idx.pkl','rb')))

vecs, words, wordidx = load_glove(res_path+'6B.100d')
vecs.shape

' '.join(words[:25])

def w2v(w): return vecs[wordidx[w]]

w2v('of')

#import imp
#imp.reload(sys)
#sys.setdefaultencoding('utf8')

tsne = TSNE(n_components=2, random_state=0)
Y = tsne.fit_transform(vecs[:500])

start=0; end=350
dat = Y[start:end]
plt.figure(figsize=(15,15))
plt.scatter(dat[:, 0], dat[:, 1])
for label, x, y in zip(words[start:end], dat[:, 0], dat[:, 1]):
    plt.text(x,y,label, color=np.random.rand(3)*0.7,
                 fontsize=14)
plt.show()



