get_ipython().magic('matplotlib inline')
import numpy as np
from glob import glob
import re
import pickle as pk

import pdb

annotation_list = glob('../../../../data/PascalVOC/VOC2005_*/Annotations/**/*.txt')

annotation_list[:10]

def parse_annotation(filename, imbase='/usr/local/python/bnr_ml/data/PascalVOC/'):
    p_inquote = re.compile('\"(.+)\"')
    p_inquote_lab = re.compile('\"(.+)\" : \"(.+)\"')
    p_coord = re.compile('\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)')
    with open(filename, 'r') as f:
        line = f.readline()
        newobj = True
        imname, objlabel, objcoord = None, None, None
        objects = []
        while line:
            try:
                if 'Image filename' in line:
                    imname = p_inquote.findall(line)[0]
                if 'Original label for object' in line:
                    objlabel = p_inquote_lab.findall(line)[0]
                    if objlabel[1] == 'none':
                        objlabel = objlabel[0]
                    else:
                        objlabel = objlabel[1]
                if 'Bounding box for object' in line:
                    objcoord = p_coord.findall(line)

                if imname is not None and objlabel is not None and objcoord is not None:
                    obj = {}
                    obj['image'] = imbase + imname
                    obj['label'] = objlabel
                    obj['p1'] = tuple([int(pix) for pix in objcoord[0]])
                    obj['p2'] = tuple([int(pix) for pix in objcoord[1]])
                    objlabel, objcoord = None, None
                    objects.append(obj)
                line = f.readline()
            except:
                pdb.set_trace()
        return objects
            

annotations = []
for annotation in annotation_list:
    annotations.append(parse_annotation(annotation))

annotations[-10:]

tmp = {}
tmp['annotations'] = annotations
annotations = tmp
with open('/usr/local/python/bnr_ml/data/PascalVOC/annotations.txt', 'wb') as f:
    pk.dump(annotations, f)

with open('/usr/local/python/bnr_ml/data/PascalVOC/annotations.txt', 'rb') as f:
    an = pk.load(f)

an['annotations'][-10:]



