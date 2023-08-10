get_ipython().run_line_magic('matplotlib', 'inline')
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='/data/COCO'
dataType='train2017'
#dataType='val2017'
#dataType='train2014'
#dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
#nms=[cat['name'] for cat in cats]
#print('COCO categories: \n{}\n'.format(' '.join(nms)))
#nms = set([cat['supercategory'] for cat in cats])
#print('COCO supercategories: \n{}'.format(' '.join(nms)))

for cat in cats:
    print('{:10s} -> {:10s}'.format(cat['supercategory'], cat['name']))

print
print('There are {} categories in total'.format(len(cats)))

#myCats = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'boat']
myCats = ['bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'boat']
myCatIds = coco.getCatIds(catNms=myCats)
myImgIds = []
for cat in myCats:
    catId  = coco.getCatIds(catNms=cat)
    imgIds = coco.getImgIds(catIds=catId)
    print('{:10s}: {:5d} images'.format(cat, len(imgIds)))
    myImgIds += imgIds
    myImgIds = list(set(myImgIds))  # remove duplicates
print('Total # of images in ground vehicles: ', len(myImgIds))

img = coco.loadImgs(myImgIds[np.random.randint(0,len(imgIds))])[0]
annIds = coco.getAnnIds(imgIds=[img['id']], catIds=myCatIds, iscrowd=None)
print(img['file_name'])
print(annIds)
print
annos = coco.loadAnns(annIds)
for anno in annos:
    #print(anno)
    x , y, w, h = anno['bbox']
    print('{} at ({}, {}, {}, {})'.format(cats[anno['category_id']-1]['name'], x, y, x+w, y+h))

# load and display image
I = io.imread('{}/images/{}/{}'.format(dataDir, dataType, img['file_name']))
# use url to load image
# I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()
plt.imshow(I); plt.axis('off')
coco.showAnns(annos)



