HOME = '/home/dueo/data/Genedata/'    #On the Datalab Server
#HOME = '/Users/oli/datasets/Genedata/ #On the MAC

import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
import os
import pandas as pd
import os
import dicom
get_ipython().magic('matplotlib inline')

images = pd.read_csv(HOME + 'Image.csv')

print(np.shape(images))
pd.DataFrame.head(images)

print(images['FileName_Hoechst'][0:3])
print(images['FileName_Nucleoli'][0:3])

row = 0
img_FileName_Hoechst = images['FileName_Hoechst'][row]
img_FileName_ER = images['FileName_ER'][row]
img_FileName_Nucleoli = images['FileName_Nucleoli'][row]
img_FileName_Golgi_Actin = images['FileName_Golgi_Actin'][row]
img_FileName_Mito = images['FileName_Mito'][row]
names = (img_FileName_Hoechst,img_FileName_ER,img_FileName_Nucleoli,img_FileName_Golgi_Actin, img_FileName_Mito)

get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (15.0, 10.0)
from PIL import Image
from matplotlib.pyplot import imshow

def showImages(row = 0):
    img_FileName_Hoechst = images['FileName_Hoechst'][row]
    img_FileName_ER = images['FileName_ER'][row]
    img_FileName_Nucleoli = images['FileName_Nucleoli'][row]
    img_FileName_Golgi_Actin = images['FileName_Golgi_Actin'][row]
    img_FileName_Mito = images['FileName_Mito'][row]
    names = (img_FileName_Hoechst,img_FileName_ER,img_FileName_Nucleoli,img_FileName_Golgi_Actin, img_FileName_Mito)

    fig = plt.figure()
    for i,name in enumerate(names):
        im = Image.open(HOME + '/BBBC022_Profiling/' + name)
        im_f = np.asarray(im, dtype='float32')
        print('Min={0} max={1}'.format(np.min(im_f), np.max(im_f)))
        ax = fig.add_subplot(2, 3, i+1)
        ax.imshow(im_f)    
showImages(0)

cells = pd.read_csv(HOME + 'Cells.csv')

pd.DataFrame.head(cells)

print(max(cells['ImageNumber']), np.shape(images))

images['FileName_ER'][873-1]

pheno = cells['Metadata_Phenotype']
np.shape(pheno)

np.histogram(pheno, bins=[0, 1, 2, 3,4,5,6])

cells[pheno == 3][0:4]

row = 72
showImages(row)

x = cells['AreaShape_Center_X']
y = cells['AreaShape_Center_Y']
(max(x), max(y))

def cutCell(x, y, d, MAL, MinAL, im_f, paint=False):
    max_x = np.shape(im_f)[1]
    max_y = np.shape(im_f)[0]
    #print('max_x={0} max_y={1}'.format(max_x,max_y))
    cx, cy = draw.circle_perimeter(x, y, 5)
    #im_f[cy, cx] = 2000
    
    xc_1 = np.cos(d)*MAL/2.0   #Determined by the Major Axis 
    yc_1 = np.sin(d)*MAL/2.0
    xc_2 = np.sin(d)*MinAL/2.0 #Determined by the Minor Axis  
    yc_2 = np.cos(d)*MinAL/2.0
    
    # Drawing the 
    cx, cy = draw.line(int(x - xc_1),int( y - yc_1),int( x + xc_1), int( y + yc_1))
    #im_f[cy, cx] = 1000
    
    xc = max(abs(xc_1), abs(xc_2))
    yc = max(abs(yc_1), abs(yc_2))
    
    #The box around the cell
    x0 = int(x - xc)
    y0 = int(y - yc)
    x1 = int(x + xc)
    y1 = int(y + yc)
    #print('i={0} x0={1} x1={2} y0={3} y1={4}'.format(i,x0,x1,y0,y1))
    if (x0 >= 0 and y0 >= 0 and x1 < max_x and y1 < max_y):
        if paint:
            im_f[y, x] = 2000
            cx, cy = draw.line(x0,y0,x0, y1)
            im_f[cy, cx] = 2000
            cx, cy = draw.line(x0,y0,x1, y0)
            im_f[cy, cx] = 2000
            cx, cy = draw.line(x1,y1,x1, y0)
            im_f[cy, cx] = 2000
            cx, cy = draw.line(x0,y1,x1, y1)
            #cx, cy = draw.line(527,314,612, 314)
            im_f[cy, cx] = 2000
            im_f[y0, x0] = 2000
            im_f[y1, x1] = 2000 
        return im_f[y0:y1,x0:x1]
    return None

from skimage import data, io, draw #http://scikit-image.org/docs/stable/
import matplotlib.pyplot as plt

row = 72
#img_FileName = images['FileName_Hoechst'][row]
img_FileName = images['FileName_Mito'][row]


im = Image.open(HOME + 'BBBC022_Profiling/' + img_FileName)
im_f = np.asarray(im, dtype='float32')
print('Values:: Min={0} max={1}'.format(np.min(im_f), np.max(im_f)))


imgCells = cells[cells['ImageNumber'] == row+1]
xs = np.asarray(imgCells['AreaShape_Center_X'])
ys = np.asarray(imgCells['AreaShape_Center_Y'])
MALs = np.asarray(imgCells['AreaShape_MajorAxisLength']) 
MinALs = np.asarray(imgCells['AreaShape_MinorAxisLength']) 
DEGs = np.asarray(imgCells['AreaShape_Orientation']) / 180.0 * np.pi
EGGs = np.asarray(imgCells['AreaShape_Eccentricity'])

max_x = np.shape(im_f)[1]
max_y = np.shape(im_f)[0]
print('max_x={0} max_y={1}'.format(max_x,max_y))

im_cut = None
for i in range(len(xs)): #range(12,13):#
    x = xs[i] #from 0 to approx 700
    y = ys[i] #from 0 to approx 500
    d = DEGs[i]
    MAL = MALs[i]
    MinAL = MinALs[i]
    im_cut_t = cutCell(x,y,d,MAL,MinAL,im_f, paint=True)
    if im_cut_t != None:
        im_cut = im_cut_t

# mark the center
for i in range(len(xs)):
    im_f[ys[i], xs[i]] = 2000
    
io.imshow(im_f)    

io.imshow(im_cut) 
x - xc, y - yc, x + xc, y + yc, np.max(xs), np.shape(im_f)

xs = np.asarray(cells['AreaShape_Center_X'])
ys = np.asarray(cells['AreaShape_Center_Y'])
MALs = np.asarray(cells['AreaShape_MajorAxisLength']) 
MinALs = np.asarray(cells['AreaShape_MinorAxisLength']) 
DEGs = np.asarray(cells['AreaShape_Orientation']) / 180.0 * np.pi
EGGs = np.asarray(cells['AreaShape_Eccentricity'])
prefix = '/Users/oli/datasets/Genedata/BBBC022_Profiling/'
PIXELS = (48,48)

numCells = len(xs) #Note that we expect even less cells found in the end since cells at the border are not used
X = np.zeros((numCells,5,PIXELS[0],PIXELS[1]), dtype='float32') 
Y = np.zeros(numCells)
cell_rows = np.zeros(numCells)

# Scales an image to (48,48) keeping the aspect ratio constant
from skimage import transform
def rectifyAndScale(res):
    m = max(np.shape(res))
    Xs = np.ones((m,m))*np.min(res)
    off_set = (np.asarray((m, m)) - np.shape(res))/(2,2)
    x1 = off_set[0]
    x2 = m-off_set[0]
    y1 = off_set[1]
    y2 = (m-off_set[1])
    d = np.shape(res)
    Xs[x1:x1+d[0],y1:y1+d[1]] = res
    return skimage.transform.resize(Xs,PIXELS)

last_im_row = -1
cell_nums = 0
for row in range(4):#range(len(xs)):
    im_row = cells['ImageNumber'][row] - 1
    if (im_row != last_im_row): # New Image
        print('New Image {0}'.format(im_row))
        last_im_row = im_row
        im_Hoechst = np.asarray(Image.open(prefix + images['FileName_Hoechst'][im_row]), dtype='float32')
        im_ER = np.asarray(Image.open(prefix + images['FileName_ER'][im_row]), dtype='float32')
        im_Nucleoli = np.asarray(Image.open(prefix + images['FileName_Nucleoli'][im_row]), dtype='float32')
        im_Golgi_Actin = np.asarray(Image.open(prefix + images['FileName_Golgi_Actin'][im_row]), dtype='float32')
        im_Mito = np.asarray(Image.open(prefix + images['FileName_Mito'][im_row]), dtype='float32')
    x = xs[row] #from 0 to approx 700
    y = ys[row] #from 0 to approx 500
    d = DEGs[row]
    MAL = MALs[row]
    MinAL = MinALs[i]
    res = cutCell(x, y, d, MAL, MinAL, im_f=im_Hoechst)
    if (res != None):
        X[cell_nums,0,:,:] = rectifyAndScale(res)
        X[cell_nums,1,:,:] = rectifyAndScale(cutCell(x, y, d, MAL, MinAL, im_f=im_ER))
        X[cell_nums,2,:,:] = rectifyAndScale(cutCell(x, y, d, MAL, MinAL, im_f=im_Nucleoli))
        X[cell_nums,3,:,:] = rectifyAndScale(cutCell(x, y, d, MAL, MinAL, im_f=im_Golgi_Actin))
        X[cell_nums,4,:,:] = rectifyAndScale(cutCell(x, y, d, MAL, MinAL, im_f=im_Mito))
        Y[cell_nums] = cells['Metadata_Phenotype'][row]
        cell_rows[cell_nums] = row
        cell_nums += 1
        #io.imshow(X)
        #print(np.shape(res))

maxIdx = cell_nums - 1
print(cell_nums)

# Note that pickle does not work in the case, since the data seems to be too large
# See: http://stackoverflow.com/questions/28503942/pickling-large-numpy-array

#import pickle
#with open('HCS_48x48.pickle', 'wb') as f:
#    pickle.dump((cell_rows[0:maxIdx],X[0:maxIdx,:,:,:],Y[0:maxIdx]), f, -1)
get_ipython().magic('ls -lh')
with open('HCS_48x48.npz', 'wb') as f:
    np.savez(f, cell_rows[0:maxIdx],X[0:maxIdx,:,:,:],Y[0:maxIdx])

X[maxIdx,1,:,:]
cell_rows[43423]

get_ipython().magic('ls -lh')



