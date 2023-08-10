import os
import glob

def read_lines(f):
    for line in f:
        if 'Image size (X x Y x C) :' in line:
            words = line.split()
            iw = int(words[8])
            ih = int(words[10])
        if 'Bounding box for object' in line:
            words = line.split()
            xmin = int(words[12][1:-1])
            ymin = int(words[13][:-1])
            xmax = int(words[15][1:-1])
            ymax = int(words[16][:-1])
            x = 1.0*(xmax+xmin)/2
            y = 1.0*(ymax+ymin)/2
            width = xmax-xmin
            heigth = ymax-ymin
            an['width'] = 1.0*width/iw
            an['height'] = 1.0*heigth/ih
            an['cx'] = 1.0*x/iw
            an['cy'] = 1.0*y/ih
            newline = '0 '+str(an['cx'])+' '+str(an['cy']) + ' '+str(an['width']) + ' '+str(an['height'])
            with open(name,'a') as w:
                w.write(newline+'\n')

# Training
for anno_fn in sorted(glob.glob('Train/annotations/*.txt')):
    iw = 0
    ih = 0
    name = ''
    an = {}
    with open(anno_fn,'r',encoding='latin-1') as f:
        name = 'Train/labels/' + anno_fn[18:-4]+'.txt'
        read_lines(f)

## Generation of the Train.txt for YOLO
for train in sorted(glob.glob('data/INRIAPerson/Train/pos/*')):
    with open('train.txt','a') as w:
                    w.write(train+'\n')

# Test
for anno_fn in sorted(glob.glob('Test/annotations/*.txt')):
    iw = 0
    ih = 0
    name = ''
    an = {}
    with open(anno_fn,'r',encoding='latin-1') as f:
        name = 'Test/labels/' + anno_fn[17:-4]+'.txt'
        read_lines(f)

## Generation of the Train.txt for YOLO
import glob
for test in sorted(glob.glob('data/INRIAPerson/Test/pos/*')):
    with open('test.txt','a') as w:
                    w.write(test+'\n')

