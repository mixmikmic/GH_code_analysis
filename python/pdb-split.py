from shutil import copyfile
from os.path import split, join
from numpy.random import random
from glob import glob

b = '/Users/nicholassofroniew/Documents/DATA-proteins/'
files = glob(b+'pdb-parsed/*.csv')

for f in files:
    head, tail = split(f)
    r = random()
    if r>.2:
        copyfile(f, join(b,'proteins/train',tail))
    elif r>.1:
        copyfile(f, join(b,'proteins/val',tail))
    else:
        copyfile(f, join(b,'proteins/test',tail))

