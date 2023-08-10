from PIL import Image
from PIL import ImageOps
import csv
import math
import matplotlib.animation as animate
import numpy as np
import matplotlib.pyplot as plt

def animateIt(filenames):
    readIms = []
    fig = plt.figure()
    for name in filenames:
        f = open("/Users/brownscholar/Desktop/codingclimatechange/SavedImages/"+name+".jpg")
        img_read = plt.imread(f)
        readIms.append([plt.imshow(img_read)])
    ani = animate.ArtistAnimation(fig, readIms, interval=1000, repeat_delay=3000, blit=False)
    plt.show()

#Where it says "Enter file names here" you need to enter the file names you want to animate, separated by commas.
animateIt(["Enter your file names here"])



