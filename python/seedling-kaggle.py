get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

from fastai.imports import *
from fastai.dataset import *
from fastai.dataloader import *
from fastai.conv_learner import *
from fastai.plots import *

PATH = '/home/ubuntu/workspace/data/iciar18/full_dataset/'

get_ipython().system('ls {PATH} | wc -l')

get_ipython().system('ls {PATH}/train/ | head')

im_is = plt.imread(f'{PATH}is007.jpg')
im_b = plt.imread(f'{PATH}b015.jpg')
im_iv = plt.imread(f'{PATH}iv087.jpg')
im_n = plt.imread(f'{PATH}n055.jpg')
ims = [im_is, im_b, im_iv, im_n]
titles=['InSitu', 'Benign', 'Invasive', 'Normal']
for k, i in enumerate(ims):
    plt.title(titles[k]+', size: '+str(i.shape[:2]))
    plt.imshow(i)
    plt.show()
    
# ims = np.stack([im_is, im_b, im_iv, im_n])
# plots(ims, rows = 2, )

import pandas as pd

dftrain = pd.read_csv(f'{PATH}train_multi.csv')
dftrain

sizes = [plt.imread(f'{PATH}{im}').shape[0] for im in dftrain[0] ]

sizes_np = np.array(sizes)
np.where(sizes_np==200)

plt.hist(sizes_np)

print(f'Mean size of images: {sizes_np.mean()}')

arch=resnet50
bs=64
sz=224

def get_val_idxs(n_per_class, val_pct, nclasses):
    v_idxs = get_cv_idxs(n_per_class, val_pct=val_pct, seed=24)
    v_next_idxs = [v_idxs + i*100 for i in range(1, nclasses)]
    v_idxs = [v_idxs] + v_next_idxs
    return np.concatenate(v_idxs)

def get_data(arch, sz, bs, val_idxs):
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_top_down, max_zoom=1.1)
    csv_fname = PATH+'microscopy_ground_truth.csv'
   
    data = ImageClassifierData.from_csv(PATH, '', csv_fname, bs,tfms,val_idxs)
    return data

val_idxs = get_val_idxs(100, 0.2, 4)

# v2 = val_idxs + 100
# v3 = val_idxs + 200
# v4 = val_idxs + 300
# val_idxs, v2, v3, v4
# vf = np.concatenate((val_idxs, v2, v3, v4))
len(val_idxs), val_idxs
# data = get_data(arch, sz, bs, vf)
# learner = ConvLearner.pretrained(arch, data, precompute=True, opt_fn=optim.Adam)
# learner.lr_find()
# learner.sched.plot()

data.bs

lrf=0.005
learner.fit(lrf, 3, cycle_len=1)

learner.fit(lrf, 3, cycle_len=1)

lr = np.array([lrf/50, lrf/10, lrf])
learner.unfreeze()
learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

learner.save('st224_res50_adam_lr005')

learner.load('st224_res50_adam_lr005')

log_preds,y = learner.TTA()

log_preds.shape, y.shape

l_p1 = np.mean(log_preds, axis=0)
l_p1.shape

# log_preds,y = learner.predict_with_targs()
preds = np.argmax(l_p1, axis=1)
probs = np.exp(log_preds[:,1])
print(accuracy(l_p1, y))
print()
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)
plot_confusion_matrix(cm, data.classes)

learner.fit(lr, 3, cycle_len=1)

learner.save('st224_res50_adam')

learner.load('st224_res50_adam')

lr = np.array([lrf/50, lrf/10, lrf])

learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

learner.save('st224_res50_adam005_wd')

learner.load('st224_res50_adam005_wd')

learner.opt_fn=optim.SGD

lr = np.array([0.001/9, 0.001/3, 0.001])
learner.fit(lr, 2, cycle_len=1, cycle_mult=2)

learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

lr = np.array([0.01/90, 0.01/30, 0.01])

log_preds,y = learner.TTA()
# log_preds = np.mean(log_preds, axis=0)
preds = np.argmax(log_preds, axis=1)
probs = np.exp(log_preds[:,1])
print(accuracy(log_preds, y), metrics.log_loss(y, np.exp(log_preds)))
print()
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)
plot_confusion_matrix(cm, data.classes)

learner.fit(lr, 3, wds=wds, use_wd_sched=True, cycle_len=1, cycle_mult=2)

lr = np.array([lrf/100, lrf/10, lrf])
learner.unfreeze()
learner.fit(lr, 3, cycle_len=1, cycle_mult=2)



learner.fit(lr, 4, cycle_len=1, cycle_mult=2)

lr = np.array([lrf/18, lrf/6, lrf])
learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

learner.save('start224_res50')



lr = np.array([lrf/9, lrf/3, lrf])
# learner.unfreeze()
learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

learner.save("32bs_224sz_res34")

learner.fit(lr, 4, cycle_len=1, cycle_mult=2) # x2

log_preds,y = learner.TTA()
metrics.log_loss(y,np.exp(log_preds)), accuracy(log_preds,y)

learner.unfreeze()
learner.load('224_res101')

learner.set_data(get_data(arch, 340, bs, val_idxs))

learner.freeze()
learner.precompute=True

learner.lr_find()
learner.sched.plot()

learner.precompute=True

lrf=0.01
learner.fit(lrf, 3, cycle_len=1)

learner.fit(lrf, 3, cycle_len=1, cycle_mult=2)

log_preds,y = learner.TTA()
accuracy(log_preds,y), metrics.log_loss(y, np.exp(log_preds))

lrf=0.01
lr = np.array([lrf/100, lrf/10, lrf])
learner.unfreeze()
learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

learner.unfreeze()
learner.load('340_res101')

y

# log_preds,y = learner.TTA()
preds = np.exp(log_preds)
print(accuracy(preds,y))
print(metrics.log_loss(y, preds))


preds = np.argmax(log_preds, axis=1)
probs = np.exp(log_preds[:,1])
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)
plot_confusion_matrix(cm, data.classes)

lrf=0.01
learner.freeze()
learner.precompute=True
learner.fit(lrf, 3, cycle_len=1)

learner.fit(lrf, 3, cycle_len=1, cycle_mult=2)

learner.unfreeze()
lr = np.array([lrf/100, lrf/10, lrf])
learner.fit(lr, 4, cycle_len=1, cycle_mult=2)

log_preds,y = learner.TTA()
accuracy(log_preds,y)

preds = np.argmax(log_preds, axis=1)
probs = np.exp(log_preds[:,1])
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)
plot_confusion_matrix(cm, data.classes)

res = learner.TTA(is_test=True)

log_preds= res[0]

preds = np.argmax(log_preds, axis=1)
probs = np.exp(log_preds[:,1])

classes = learner.data.classes

classes = [c.replace('_', ' ') for c in cle]

fnames = [f.split('/')[-1] for f in learner.data.test_ds.fnames]

res = {}
for k, f in enumerate(fnames):
    res[f] = preds[classesk]

import pandas as pd
df = pd.DataFrame.from_dict(res, orient='index')

df.to_csv('pred.csv', header=False)

