get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

from fastai.imports import *
from fastai.dataset import *
from fastai.dataloader import *
from fastai.conv_learner import *
from fastai.plots import *

PATH = '/home/ubuntu/workspace/data/iciar18/full_dataset/'

im_is = plt.imread(f'{PATH}is001.jpg')
im_b = plt.imread(f'{PATH}b001.jpg')
im_iv = plt.imread(f'{PATH}iv001.jpg')
im_n = plt.imread(f'{PATH}n001.jpg')
ims = np.stack([im_is, im_b, im_iv, im_n])
plots(ims, rows = 2, titles=['InSitu', 'Benign', 'Invasive', 'Normal'])
print(im_n.shape)
# plt.imshow(im); plt.show()

def get_val_idxs(n_per_class, val_pct, nclasses):
    v_idxs = get_cv_idxs(n_per_class, val_pct=val_pct, seed=24)
    v_next_idxs = [v_idxs + i*100 for i in range(1, nclasses)]
    v_idxs = [v_idxs] + v_next_idxs
    return np.concatenate(v_idxs)

def get_data(csv_fname, arch, sz, bs, val_idxs, aug=transforms_top_down):
    tfms = tfms_from_model(arch, sz, aug_tfms=aug, max_zoom=1.05)
    data = ImageClassifierData.from_csv(PATH, '', f'{PATH}{csv_fname}', bs,tfms,val_idxs)
    return data

def plot_cm(classes, y, preds):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, preds)
    plot_confusion_matrix(cm, classes)
    
def evaluate(learner, aug=False):
    aug_logs = None
    log_preds,y = None, None
    if aug:
        aug_logs,y = learner.TTA()
        log_preds = np.mean(aug_logs, axis=0)
    else: 
        log_preds,y = learner.predict_with_targs()
    print(accuracy(log_preds,y))
    if aug:
        for l in aug_logs:
            print(accuracy(l,y))
    preds = np.argmax(log_preds, axis=1)
    plot_cm(learner.data.classes, y, preds)
    return aug_logs, y

crossvalids = np.load('cv5folds.txt.npy')
cross_idx = crossvalids[0]

def get_crossv_idxs(cv, nclasses):
    cv = np.array(cv)
    v_next_idxs = [cv + i*100 for i in range(1, nclasses)]
    v_idxs = [cv] + v_next_idxs
    return np.concatenate(v_idxs)

cv_idxs = get_crossv_idxs(cross_idx, 4)
cv_idxs.shape, cv_idxs

arch=resnet50
bs=10
sz=299
aug = [RandomRotateXY(10), RandomDihedralXY(), RandomFlipXY()]
val_idxs = get_crossv_idxs(cross_idx, 4)
data = get_data('train_multi.csv', arch, sz, bs, val_idxs, aug=aug)
learner = ConvLearner.pretrained(arch, data,precompute=False)

learner.lr_find()
learner.sched.plot()

learner.sched.plot(1)

lrf = 0.01
learner.fit(lrf, 3, cycle_len=1)

# learner.fit(lrf, 3, cycle_len=1)

learner.unfreeze()
lr = np.array([lrf/25., lrf/5., lrf])
learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

learner.save('bin_carci_5cv1_85')

learner.load('bin_carci_5cv1_85')

learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

learner.save('bin_carci_5cv1_90')

#learner.load('bin_carci_sz_400_cv1_88')
learner.load('bin_carci_5cv1_90')

a = evaluate(learner, True)

learner.fit(lr, 3, cycle_len=1, cycle_mult=2, cycle_save_name='bin_carci_5cv1')

learner.load('bin_carci_5cv1_cyc_0_91')
# a = evaluate(learner, True)

learner.save('bin_carci_5cv1_91')

learner.load('bin_carci_5cv1_91')

a = evaluate(learner, True)

learner.fit(lr, 3, cycle_len=1, cycle_mult=2, cycle_save_name='bin_carci_5cv1')

learner.save('bin_carci_5cv1_97')
# a = evaluate(learner,False)

learner.load('bin_carci_5cv1_97')

learner.fit(lr, 3, cycle_len=1, cycle_mult=3)

learner.save('bin_carci_5cv1_96_good_loss')

learner.load('bin_carci_5cv1_96_good_loss')

learner.fit(lr, 2, cycle_len=1, cycle_mult=3, cycle_save_name='bin_carci_5cv1')

learner.load('bin_carci_5cv1_97_good_loss')
a=evaluate(learner,False)

learner.fit(lr/10, 2, cycle_len=1, cycle_mult=3, cycle_save_name='bin_carci_5cv1')

learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

learner.save('bin_carci_5cv1_95')

learner.load('bin_carci_5cv1_95')

learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

arch=resnet50
bs=10
sz=299
aug = [RandomRotateXY(10), RandomDihedralXY(), RandomFlipXY()]
val_idxs = get_crossv_idxs(cross_idx, 2)
data = get_data('train_norbe.csv', arch, sz, bs, val_idxs, aug=aug)
learner = ConvLearner.pretrained(arch, data, precompute=False)
learner.lr_find()
learner.sched.plot()

learner.sched.plot(1)

lrf = 0.01
learner.fit(lrf, 3, cycle_len=1)

learner.unfreeze()
lr = np.array([lrf/25., lrf/5., lrf])
learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

learner.save('bin_norbe_bs10_96')

learner.load('bin_norbe_bs10_96')

a = evaluate(learner, False)

learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

learner.save('bin_norbe_bs10_98')

learner.load('bin_norbe_bs10_98')

a=evaluate(learner, False)

a=evaluate(learner, True)

learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

learner.save('bin_norbe_bs10_100')

learner.load('bin_norbe_bs10_100')

a=evaluate(learner)

arch=resnet50
bs=10
sz=299
aug = [RandomRotateXY(10), RandomDihedralXY(), RandomFlipXY()]
val_idxs = get_val_idxs(100, 0.25, 2)
data = get_data('train_invis.csv', arch, sz, bs, val_idxs, aug=aug)
learner = ConvLearner.pretrained(arch, data, precompute=False)
learner.lr_find()
learner.sched.plot()

learner.sched.plot(1)

lrf = 0.01
learner.fit(lrf, 3, cycle_len=1)

learner.unfreeze()
lr = np.array([lrf/25., lrf/5., lrf])
learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

a=evaluate(learner, False)

learner.save('bin_invis_bs10_96')

learner.load('bin_invis_bs10_96')

learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

a=evaluate(learner,True)

plot_cm(learner.data.classes, a[1], np.argmax(a[0][2],axis=1))

learner.save('bin_invis_bs10_96_bis')

learner.load('bin_invis_bs10_96_bis')

learner.fit(lr, 3, cycle_len=1, cycle_mult=2)

learner.fit(lr, 3, cycle_len=1, cycle_mult=3)



log_probs, y = learner.predict_with_targs()
# pr = Prob(image_class == 'Non_Carcinoma')
# if pr < 0.5 => image_class = Carcinoma
# else: image_class = Non_Carcinoma
probs = np.exp(log_probs[:,1]) 
preds = np.argmax(log_probs, axis=1)
learner.data.classes

def rand_by_mask(mask, n=4):
    m = np.where(mask)[0]
    print(f'nb images :{len(m)}')
    return np.random.choice(m, n, replace=False)
def rand_by_correct(is_correct, n=4): return rand_by_mask((preds == data.val_y)==is_correct, n)

def plot_val_with_title(idxs, title, rows=1):
    imgs = np.stack([data.val_ds[x][0] for x in idxs])
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(data.val_ds.denorm(imgs), rows=rows, titles=title_probs)

def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])

def load_img_id(ds, idx): return np.array(PIL.Image.open(PATH+ds.fnames[idx]))

def plot_val_with_title(idxs, title, rows=1):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
    title_probs = [(probs[x], data.val_ds.fnames[x]) for x in idxs]
    print(title)
    return plots(imgs, rows=rows, titles=title_probs, figsize=(16,8))

# 1. A few correct labels at random

plot_val_with_title(rand_by_correct(True), "Correctly classified", rows=2)

plot_val_with_title(rand_by_correct(False, 1), "InCorrectly classified", rows=1)

def most_by_mask(mask, mult):
    idxs = np.where(mask)[0]
    print(f'nb images :{len(idxs)}')
    return idxs[np.argsort(mult * probs[idxs])[:4]]

def most_by_correct(y, is_correct): 
    mult = -1 if (y==1)==is_correct else 1
    return most_by_mask((preds == data.val_y)==is_correct & (data.val_y == y), mult)
plot_val_with_title(most_by_correct(0, True), "Most correct Benign",rows=2)

plot_val_with_title(most_by_correct(0, False), "Most INCORRECTS Benign",rows=1)

plot_val_with_title(most_by_correct(1, True), "Most correct Normal", rows=1)

plot_val_with_title(most_by_correct(1, False), "Most INCORRECT Normal")

most_uncertain = np.argsort(np.abs(probs -0.5))[:4]
plot_val_with_title(most_uncertain, "Most uncertain predictions")

