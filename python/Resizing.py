get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt

from PIL import Image
import numpy as np
from scipy.misc import imresize
import glob, threading, bcolz, os, time, pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from vgg16 import VGG16
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras.callbacks import History 
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def limit_mem():
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

def clear_mem():
    sess = K.get_session()
    sess.close()
    limit_mem()
    return
    
dpath = '../../fastdata/dl1data/dogscats/'
desired_shp = 224

limit_mem()

fnames = list(glob.iglob(dpath + 'sample/train/*/*.jpg'))
orig = Image.open(fnames[2]); print(orig.size)

orig_arr = np.array(orig)
plt.imshow(orig_arr)

resized = imresize(orig_arr, (desired_shp,desired_shp))
plt.imshow(resized)

def resize_center_crop(im_arr, desired_shp = 224):
    x,y,_ = im_arr.shape
    ratio = 1.0 * x/y
    if x < y:
        newx, newy = desired_shp, desired_shp/ratio
    elif y < x:
        newx, newy = desired_shp * ratio, desired_shp
    else :
        newx, newy = desired_shp,desired_shp
    newshape = (int(newx), int(newy))
    
    #Resize based shortest side to 'desired_shp'
    im_arr = imresize(im_arr, newshape)
    
    #center crop so both side are equal length
    left, right = (newx - desired_shp)/2, (newx + desired_shp)/2
    top, bottom = (newy - desired_shp)/2, (newy + desired_shp)/2
    return im_arr[left:right,top:bottom]

plt.imshow(orig_arr)

center_cropped = resize_center_crop(orig_arr)
plt.imshow(center_cropped)

orig_arr = np.array(orig)
plt.imshow(orig_arr)

def mk_square(img, desired_shp=224):
    x,y,_ = img.shape
    maxs = max(img.shape[:2])
    y2=(maxs-y)//2
    x2=(maxs-x)//2
    arr = np.zeros((maxs,maxs,3), dtype=np.float32)
    arr[np.floor(x2):np.floor(x2)+x,np.floor(y2):np.floor(y2)+y] = img
    return imresize(arr, (desired_shp,desired_shp))

resized_square = mk_square(orig_arr)
plt.imshow(resized_square)

# #Setup data dirs
# if not os.path.exists(f'{dpath}/resized/'): os.mkdir(f'{dpath}/resized/')
# for data_group in ['train', 'test1', 'valid']:
#     for method in fn_info.keys():
#         if not os.path.exists(f'{dpath}/resized/{method}/'): 
#             os.mkdir(f'{dpath}/resized/{method}/')


def resize_img_1(i ,desired_shp=224):
 '''Squashes to square, regardless of aspect ratio'''
 orig_arr = np.array(Image.open(fnames[i]))
 return imresize(orig_arr, (desired_shp,desired_shp))

def resize_img_2(i ,desired_shp=224):
 '''Resizes shortest side to 224, then centercrops'''
 orig_arr = np.array(Image.open(fnames[i]))
 return resize_center_crop(orig_arr, desired_shp)

def resize_img_3(i ,desired_shp=224):
 '''Converts to a square image with black borders to retain aspect ratio'''
 orig_arr = np.array(Image.open(fnames[i]))
 return mk_square(orig_arr, desired_shp)

fn_info = {
 'squash':resize_img_1,
 'center_crop':resize_img_2,
 'black_border':resize_img_3
 }

#Create pre-allocated memory to speed things up
tl = threading.local()
tl.place = np.zeros((desired_shp,desired_shp,3), 'uint8')

def app_img(r):
    tl.place[:] = np.array(r)[0:desired_shp, 0:desired_shp]
    arr.append(tl.place) 


#For each data set, get filenames and parallel resize. 
for data_group in ['train', 'test1', 'valid']:
    fnames = list(glob.iglob(dpath + f'/{data_group}/*/*.jpg'))
    for method in fn_info.keys():
        fn = fn_info[method]
        bc_path = f'{dpath}/resized/{method}/{method}_{data_group}_{desired_shp}_r.bc'
        arr = bcolz.carray(np.empty((0, desired_shp, desired_shp, 3), 'float32'), 
                       chunklen=32, mode='w', rootdir=bc_path)
        step, n =6400, len(fnames)
        for i in range(0, n, step):
            with ThreadPoolExecutor(max_workers=16) as execr:
                res = execr.map(fn, range(i, min(i+step, n)))
                for r in res: app_img(r)
            arr.flush()

#For each data set, create class labels we can use in training
def get_classes(data_group):
    fnames = list(glob.iglob(dpath + f'/{data_group}/*/*.jpg'))
    fnames = [f.split('/')[-1].split('.')[0] for f in fnames]
    fnames = [1 if f == 'cat' else 0 for f in fnames ]
    return to_categorical(fnames)

for data_group in ['train', 'test1', 'valid']:
    class_path = f'{dpath}/resized/{data_group}_classes.bc'
    save_array(class_path, get_classes(data_group))

# Generate VGG Convolutional Features
methods = ['center_crop', 'squash', 'black_border']
for method in methods:
    trn = load_array(f'{dpath}/resized/{method}/{method}_train_{desired_shp}_r.bc')
    val = load_array(f'{dpath}/resized/{method}/{method}_valid_{desired_shp}_r.bc')
    tst = load_array(f'{dpath}/resized/{method}/{method}_test1_{desired_shp}_r.bc')

    print(trn.shape, tst.shape, val.shape)

    vgg = VGG16(include_top=False, weights='imagenet')

    conv_val_ftrs = vgg.predict(val)
    conv_trn_ftrs = vgg.predict(trn)
    conv_tst_ftrs = vgg.predict(tst)

    save_array(f'{dpath}/resized/{method}/{method}_val_{desired_shp}_vggout.bc', conv_val_ftrs)
    save_array(f'{dpath}/resized/{method}/{method}_trn_{desired_shp}_vggout.bc', conv_trn_ftrs)
    save_array(f'{dpath}/resized/{method}/{method}_tst_{desired_shp}_vggout.bc', conv_tst_ftrs)

    del trn, val, tst
    del vgg

def build_model(inp_shape, p=.5):
    inp = Input(shape=inp_shape)
    x = Flatten(name='flatten')(inp)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = BatchNormalization()(x)
    x = Dropout(p)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = BatchNormalization()(x)
    x = Dropout(p)(x)
    x = Dense(4096, activation='relu', name='fc3')(x)
    x = BatchNormalization()(x)
    x = Dropout(p/2)(x)
    preds = Dense(2, activation='softmax', name='predictions')(x)

    model = Model(inp, preds)
    model.compile(Adam(lr=1E-4), 'binary_crossentropy', metrics=['accuracy'])
    return model

def generate_batches(data_arr, class_arr, batch_size):
    '''Custom generator to pass some data to the model'''
    steps = data_arr.shape[0] // batch_size
    rand_idx = np.random.permutation(range(len(data_arr)))
    data_arr = data_arr[rand_idx]
    class_arr = class_arr[rand_idx]
    while 1:
        for step in range(steps):
            data = data_arr[step * batch_size:(step+1) * batch_size]
            classes = class_arr[step * batch_size:(step+1) * batch_size]
        yield (data, classes)


trn_cls = load_array(f'{dpath}/resized/train_classes.bc')
val_cls = load_array(f'{dpath}/resized/valid_classes.bc')



model = None
methods = ['center_crop', 'squash', 'black_border']
histories = {m:[] for m in methods} #Storage object for histories
for iteration in range(20):
    print(f"Training pass {iteration}")
    for method in methods:
        clear_mem()
        print(f'Starting {method}')
        conv_val_ftrs = load_array(f'{dpath}/resized/{method}/{method}_val_{desired_shp}_vggout.bc')
        conv_trn_ftrs = load_array(f'{dpath}/resized/{method}/{method}_trn_{desired_shp}_vggout.bc')
        conv_tst_ftrs = load_array(f'{dpath}/resized/{method}/{method}_tst_{desired_shp}_vggout.bc')

        batch_size, n_epoch = 64, 30
        n_trn, n_val = 10 * batch_size, 5 * batch_size
        trn_batches = generate_batches(conv_trn_ftrs, trn_cls, batch_size)
        val_batches = generate_batches(conv_val_ftrs, val_cls, batch_size)
        model = build_model(conv_trn_ftrs.shape[1:])
        history_cb = History() #keras callback for storing history of training
        results = model.fit_generator(trn_batches, samples_per_epoch=n_trn, nb_epoch=n_epoch,
                                      validation_data=val_batches, nb_val_samples=n_val,
                                      callbacks = [history_cb], verbose=0)
        histories[method].append(results.history)
        del conv_trn_ftrs; del conv_tst_ftrs; del conv_val_ftrs
        del model

#Save all results
history_path = f'{dpath}/resized/histories_batchsize64_epochs30.pkl'
pickle.dump(histories, open(history_path, 'wb'))    

history_path = f'{dpath}/resized/histories_batchsize64_epochs30.pkl'
histories = pickle.load(open(history_path, 'rb'))

fig = plt.figure(figsize=(12,5))
colors = {'squash':'#ff4c4c', 'center_crop':'#7f7fff', 'black_border':'#66b266'}
for method in methods:
    r = histories[method][5]
    c = colors[method]
    plt.plot(r['val_acc'], label=f'Validation Accuracy {method}', c=c, alpha=.6)
plt.legend()
plt.title('Validation Accuracy for Multiple Image Resizing Methods')
plt.show()
fig.savefig('./validation_acc_1pass.png')

fig = plt.figure(figsize=(12,5))
colors = {'squash':'#ff4c4c', 'center_crop':'#7f7fff', 'black_border':'#66b266'}
for method in methods:
    results = histories[method]
    c = colors[method]
    for i in range(len(results)):
        plt.plot(results[i]['val_acc'], label=f'Validation Accuracy {method}' if i==0 else ""
                 , c=c, alpha=.6)
plt.legend()
plt.title('Validation Accuracy for Multiple Image Resizing Methods')
plt.show()
fig.savefig('./validation_acc_20pass.png')

plt.legend()

fig = plt.figure(figsize=(12,5))
colors = {'squash':'#ff4c4c', 'center_crop':'#7f7fff', 'black_border':'#66b266'}
for method in methods:
    results = histories[method]
    c = colors[method]
    for r in results:
        plt.plot(r['val_loss'], label=f'Validation Loss {method}', c=c, alpha=.6)
plt.legend()
plt.title('Validation Loss for Multiple Image Resizing Methods')
plt.show()
fig.savefig('./validation_loss_20pass.png')

