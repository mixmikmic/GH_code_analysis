import string 
from collections import defaultdict

import cv2
import numpy as np 
from scipy import misc
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import tqdm

FONT = cv2.FONT_HERSHEY_DUPLEX
SIZE = 15 #size of sides of ascii .PNG (square)

#file names 
input_image = 'Florence1.jpg'
output_image = 'florence_ascii.jpg'
output_gif = 'florence_ascii.gif'

'''map ascii characters with the percentage of canvas they fill 
with color -- in a sortable dictionary (ascii_percent)'''

chars = string.ascii_letters + string.digits #all the ascii characters 
ascii_percent = defaultdict(int)

for i in range(len(chars)):
    img = np.ones(shape=(28, 28))
    cv2.putText(img, chars[i], (3, 25), FONT, 1, 0, 1)
    ascii_percent[chars[i]] = 100 - (np.sum(img)/(28**2)*100)
    
    img = misc.imresize(img, (SIZE, SIZE))
    if chars[i] in string.ascii_lowercase: 
        cv2.imwrite('chars/' + chars[i] + '_.png', img)
    else: 
        cv2.imwrite('chars/' + chars[i] + '.png', img)
    
    
#     plt.figure(figsize=(1, 1)); plt.imshow(img, cmap='gray'); plt.show()

ascii_percent = sorted(ascii_percent.items(), key=lambda k_v: k_v[1]) #sort the dict by percent

'''scale percentages [0-255]'''
mini = ascii_percent[0][1]
maxi = ascii_percent[-1][1]

def scaler(val):
    return int(((val - mini )/ (maxi-mini))*255)

'''build look up table for pixel value [0-255] and ascii char square image'''

ascii_lookup = {}
for i in range(len(ascii_percent)):

    if ascii_percent[i][0] in string.ascii_lowercase: 
        name = 'chars/' + ascii_percent[i][0] + '_.png'
    else: 
        name = 'chars/' + ascii_percent[i][0] + '.png'
    gray =  cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2GRAY)
    ascii_lookup[scaler(ascii_percent[i][1])] = gray

'''setup image conversion '''
img = cv2.GaussianBlur(cv2.imread(input_image), (7, 7), 0) #open and denoise image 
# img = misc.imresize(img, 0.8) #make a little smaller 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #grayscale 

#init output image (larger for ascii image replacement)
out  = np.ones(shape = tuple([i*SIZE for i in (img.shape)]))

'''pointwise pixel replacement 
iterate through the pixels in the image and map the 
pixel value to a corresponding ASCII character which fills a 
similar percentage of its canvas '''

for x in tqdm.tqdm(range(img.shape[0])):
    for y in range(img.shape[1]):
        pix = img[x][y]
        #find the closest ascii PNG to the pixel value
        closest = min(ascii_lookup, key=lambda x:abs(x-pix))
        ascii_arr = ascii_lookup[closest]
        
        out_x, out_y = x*SIZE, y*SIZE
        out[out_x:out_x+SIZE, out_y:out_y+SIZE] = ascii_arr
cv2.imwrite(output_image, out)      

'''make a gif!'''

import glob
import imageio
from natsort import natsorted

#for video 
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# writer = cv2.VideoWriter('tester.avi', fourcc, 15, out.shape[::-1], False)
#     writer.write(resized)
# writer.release()

#create gif 
ascii_img = out.copy()
with imageio.get_writer(output_gif, mode='I', fps=30) as writer:   
    for i in tqdm.tqdm(range(200)): 
        a_shp = ascii_img.shape
        a_shp5 = [int(a*0.01) for a in a_shp]
        a_shp95 =[int(a*0.99) for a in a_shp]
        ascii_img = ascii_img[a_shp5[0]:a_shp95[0],
                              a_shp5[1]:a_shp95[1]].astype('u1')
        resized = misc.imresize(ascii_img, img.shape)

        writer.append_data(resized)
        cv2.imwrite('gif/'+str(i)+'.png',  resized)

