import json
from pprint import pprint
import numpy as np
Settings = json.load(open('settings.txt'))
pprint(Settings)
print("")
from pak.datasets.MOT import MOT16
from os import makedirs, listdir
from os.path import join, isfile, isdir, exists, splitext
import subprocess

deepmatch_loc = Settings['deepmatch']
assert(isfile(deepmatch_loc))

root = Settings['data_root']
mot16 = MOT16(root)

from math import ceil, floor

img_loc = mot16.get_test_imgfolder("MOT16-02")

frames = sorted([join(img_loc, f) for f in listdir(img_loc)                   if f.endswith('.jpg')])

def deepmatch(img1, img2):
    args = (deepmatch_loc, img1, img2, '-downscale', '3', '-nt', '16')
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    B = np.fromstring(popen.stdout.read(), sep=' ')
    n = B.shape[0]
    assert(floor(n) == ceil(n))
    assert(floor(n/6) == ceil(n/6))
    B = B.reshape((int(n/6), 6))
    return B


TOTAL = []
for i in range(len(frames)):
    curr_frame = []
    for j in range(i+1, min(i+30, len(frames))):
        print("solve " + str(i) + " -> " + str(j))
        M = deepmatch(frames[i], frames[j])
        curr_frame.append(M)
    
    TOTAL.append(curr_frame)

TOTALnp = np.array(TOTAL)
np.save('MOT16_02.npy', TOTALnp)

