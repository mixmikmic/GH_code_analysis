import os
import sys
import time
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib tk')

plt.ion()
plt.rcParams["image.cmap"] = "gist_gray"

path1 = "images"
path2 = "dot"
dpath = os.path.join(path1, path2)
flist = [os.path.join(dpath, i) for i in 
         sorted(os.listdir(os.path.join(dpath)))]

for fname in flist:
    print(fname)

imgs = np.array([nd.imread(i) for i in flist])
nimg, nrow, ncol = imgs.shape[0:3]
xs = 5
ys = 2 * xs * float(nrow) / float(ncol)

plt.close(0)
fig0, ax0 = plt.subplots(2, 1, num=0, figsize=[xs, ys])
fig0.subplots_adjust(0, 0, 1, 1, 0, 0)
[i.axis("off") for i in ax0]
im0a = ax0[0].imshow(imgs[0])
fig0.canvas.draw()

for img in imgs:
    im0a.set_data(img)
    fig0.canvas.draw()
    time.sleep(0.02)

im0b = ax0[1].imshow(imgs[1].mean(-1) - imgs[0].mean(-1))
fig0.canvas.draw()

for ii in range(1, nimg):
    im0a.set_data(imgs[ii])
    im0b.set_data(imgs[ii].mean(-1) - imgs[ii-1].mean(-1))
    fig0.canvas.draw()
    time.sleep(0.02)

im0b.set_clim(0, 128)

for ii in range(1, nimg):
    im0a.set_data(imgs[ii])
    im0b.set_data(np.abs(imgs[ii].mean(-1) - 
                         imgs[ii-1].mean(-1)))
    fig0.canvas.draw()
    time.sleep(0.02)

dimg = np.zeros([nrow, ncol])
im0b.set_clim(0, 1)

for ii in range(1, nimg):
    dimg[:, :] = np.abs(imgs[ii].mean(-1) - 
                        imgs[ii-1].mean(-1))
    im0a.set_data(imgs[ii])
    im0b.set_data(dimg > 40)
    fig0.canvas.draw()
    time.sleep(0.02)

mimg = imgs.mean(0)

im0a.set_data(mimg.clip(0, 255).astype(np.uint8))
fig0.canvas.draw()

for ii in range(1, nimg):
    im0a.set_data(imgs[ii])
    im0b.set_data(np.abs(1.0 * imgs[ii] - mimg)                   .clip(0, 255).astype(np.uint8))
    fig0.canvas.draw()
    time.sleep(0.02)

im0b.set_clim([0, 255])

for ii in range(1, nimg):
    im0a.set_data(imgs[ii])
    im0b.set_data(np.abs(1.0 * imgs[ii] - mimg).max(-1))
    fig0.canvas.draw()
    time.sleep(0.02)

fig1, ax1 = plt.subplots(num=1)
ax1.hist(np.log10(np.abs(1.0 * imgs - mimg)                   .max(-1).flatten() + 1.0), bins=255)
fig1.canvas.draw()

thr = 30
im0b.set_clim([0, 1])

for ii in range(1, nimg):
    im0a.set_data(imgs[ii])
    im0b.set_data(np.abs(1.0 * imgs[ii] - mimg).max(-1) > thr)
    fig0.canvas.draw()
    time.sleep(0.02)

bdilation = nd.morphology.binary_dilation
berosion = nd.morphology.binary_erosion

thr = 30
im0b.set_clim([0, 1])
fgr = np.zeros([nrow, ncol], dtype=int)

for ii in range(1, nimg):
    fgr[:, :] = bdilation(berosion(np.abs(1.0 * imgs[ii] - 
                                          mimg).max(-1) > thr), 
                          iterations=2)
    im0a.set_data(imgs[ii])
    im0b.set_data(fgr)
    fig0.canvas.draw()
    time.sleep(0.02)

col_mask = (np.arange(nrow * ncol)             .reshape(nrow, ncol) % ncol) < 250

thr = 30
im0b.set_clim([0, 1])
fgr = np.zeros([nrow, ncol], dtype=int)

for ii in range(1, nimg):
    fgr[:, :] = bdilation(berosion(np.abs(1.0 * imgs[ii] - 
                                          mimg).max(-1) > thr), 
                          iterations=2)
    im0a.set_data(imgs[ii])
    im0b.set_data(fgr * col_mask)
    fig0.canvas.draw()
    time.sleep(0.02)

count = ax0[1].text(0, 0, "# of cars: ", va="top",
                    fontsize=20, color="white")
fig0.canvas.draw()

meas_label = nd.measurements.label

thr = 30
sz_thr = 20 # only count objects greater than a size threshold
im0b.set_clim([0, 1])
fgr = np.zeros([nrow, ncol], dtype=int)

for ii in range(1, nimg):
    fgr[:, :] = bdilation(berosion(np.abs(1.0 * imgs[ii] - 
                                          mimg).max(-1) > thr), 
                          iterations=2)
    labs = meas_label(fgr * col_mask)
    ncar = sum([1 * ((labs[0] == lab).sum() > sz_thr) for lab 
                in range(1, labs[1] + 1)])
    im0a.set_data(imgs[ii])
    im0b.set_data(fgr * col_mask)
    count.set_text("# of cars: {0}".format(ncar))
    fig0.canvas.draw()
    time.sleep(0.02)



