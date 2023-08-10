# Assumes Python 2.7
from IPython.display import display, Image
import numpy as np
from scipy import ndimage
import os

def renameImages(folder):
    '''Folder = name of folder containing images. Function renames all images starting from 0.jpg'''
    path = os.path.abspath(folder)
    print(path)
    names = []
    for image in os.listdir(path):
        if image.startswith(".") or image.startswith("Icon"):
            pass
        else:
            names.append(image)
    print(names)
    prefix = 0
    for name in names:
        fullname = os.path.join(path, name) 
        newname = str(prefix) + '.jpg'
        fullnewname = os.path.join(path, newname)
        print(fullnewname)
        os.rename(fullname, fullnewname)
        prefix += 1
    return

renameImages('SFTestImages')
renameImages('SF')

