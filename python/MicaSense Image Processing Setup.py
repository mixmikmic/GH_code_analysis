import cv2 #openCV
import exiftool
import numpy as np
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')

exiftoolPath = None
if os.name == 'nt':
    exiftoolPath = 'C:/exiftool/exiftool.exe'
with exiftool.ExifTool(exiftoolPath) as exift:
    print('Exiftool works!')

# Use pyplot to load an example image and display with matplotlib
imageName = os.path.join('.','data','0000SET','000','IMG_0000_4.tif')
imageRaw=plt.imread(imageName).T
plt.imshow(imageRaw.T, cmap='gray')
plt.show()

print('Success! Now you are ready for Part 1 of the tutorial.')

