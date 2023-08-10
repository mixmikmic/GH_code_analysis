get_ipython().run_line_magic('matplotlib', 'inline')

import fnmatch
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
from sklearn.decomposition import PCA

lfw_path = "../data/lfw_funneled"

images = []
for root, dirnames, filenames in os.walk(lfw_path):
    for filename in fnmatch.filter(filenames, '*.jpg'):
        images.append(os.path.join(root, filename))

n = len(images)
print("loaded %d face images" % n)

img = [Image.open(images[int(n*random.random())]) for i in range(8)]
img = np.concatenate([np.array(im.getdata()).reshape((im.size[0], im.size[1], 3)) for im in img ], axis=1)
plt.figure(figsize=(16,2))
plt.imshow(Image.fromarray(img.astype('uint8')))

w = h = 100

X = np.zeros((n, w*h*3))
for i, img in tqdm(enumerate(images)):
    im = Image.open(img)
    im = im.resize((w, h))#.convert('L')
    pixels = list(im.getdata())
    X[i, :] = np.array(pixels).flatten()

n_components = 100
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
pca.fit(X)

Xp = pca.transform(X)
print(Xp)

Xt = pca.inverse_transform(Xp)
Xt = np.clip(Xt, 0, 255)

img_pairs = [[X[idx].reshape((w,h,3)), Xt[idx].reshape((w,h,3))] for idx in np.random.randint(n, size=8)]
img_all = np.concatenate([ np.concatenate([p[0], p[1]]) for p in img_pairs], axis=1)
plt.figure(figsize=(16,4))
plt.imshow(Image.fromarray(img_all.astype('uint8')))

Y = [ np.random.normal(size=n_components) for i in range(8) ] 
Yt = [ np.clip(pca.inverse_transform(y), 0, 255) for y in Y ]
Yt = np.concatenate([ yt.reshape((w,h,3)) for yt in Yt ], axis=1)
plt.figure(figsize=(16,2))
plt.imshow(Image.fromarray(Yt.astype('uint8')))

for i in range(4):
    xp1, xp2 = Xp[int(n*random.random())], Xp[int(n*random.random())]
    Z = [[(1.0-r)*i1 + r*i2 for i1, i2 in zip(xp1, xp2)] for r in np.linspace(0,1,8)]
    Y = np.concatenate([ np.clip(pca.inverse_transform(z), 0, 255).reshape((w,h,3)) for z in Z ], axis=1)
    plt.figure(figsize=(16,2))
    plt.imshow(Image.fromarray(Y.astype('uint8')))

npca = [2000, 1000, 500, 100, 50, 10, 2, 1]
Xt = []
for p in npca:
    print("computing PCA for %d components"%p)
    pca = PCA(n_components=p, svd_solver='randomized', whiten=True)
    pca.fit(X)
    Xp = pca.transform(X)
    Xt.append(np.clip(pca.inverse_transform(Xp), 0, 255))

idxs = np.random.randint(n, size=4)
for i,idx in enumerate(idxs):
    img_orig = X[idx].reshape((w,h,3))
    img_reconstructed = np.concatenate([Xt[p][idx].reshape((w,h,3)) for p in range(len(npca))], axis=1)
    plt.figure(figsize=(2,2))
    plt.imshow(Image.fromarray(img_orig.astype('uint8')))
    plt.figure(figsize=(16,2))
    plt.imshow(Image.fromarray(img_reconstructed.astype('uint8')))

import scipy.misc
import os

cols, rows = 10, 4
nf = 24  # num frames per transitions
ni = 4   # num transitions

folder='temp'
os.makedirs(folder)
os.makedirs('%s/composite'%folder)
print("generating frames for individual faces")
for i in tqdm(range(cols)):
    for j in range(rows):
        idx = np.random.randint(n, size=ni)
        idx = np.append(idx, idx[0])
        f = 1
        for k in range(ni):
            xp1, xp2 = Xp[idx[k]], Xp[idx[k+1]]
            Z = [[(1.0-r)*i1 + r*i2 for i1, i2 in zip(xp1, xp2)] for r in np.linspace(0,1,nf)]
            Y = [ np.clip(pca.inverse_transform(z),0,255).reshape((w,h,3)) for z in Z ]
            for y in Y:
                scipy.misc.imsave('%s/face%02d_%02d_frame%04d.png'%(folder,i,j,f), y.astype('uint8'))
                f+=1
offset = np.random.randint(nf*(ni-1), size=cols*rows)
print("compositing all the faces into one video")
for f in tqdm(range(nf*ni)):
    img = Image.new('RGB',(w * cols, h * rows))
    for i in range(cols):
        for j in range(rows):
            fidx = 1+(f + offset[i*rows+j])%(nf*ni)
            im = Image.open('%s/face%02d_%02d_frame%04d.png'%(folder,i,j,fidx))
            img.paste(im, (i*w, j*h))
    img = img.resize((int(1.2*img.size[0]), int(1.2*img.size[1]))) #upscale just a bit
    scipy.misc.imsave('temp/composite/frame%04d.png'%f, img)
cmd = "ffmpeg -i %s/composite/frame%%04d.png -c:v libx264 -pix_fmt yuv420p composite.mp4"%folder
print(cmd)
os.system(cmd)
os.system("rm -rf %s"%folder)
print("done!")

