# color histogram 

from PIL import Image
from operator import itemgetter
im = Image.open("captcha2.jpeg")
im = im.convert("P")
his = im.histogram()

values = {}

for i in range(256):
    values[i] = his[i]
    
for j,k in sorted(values.items(), key=itemgetter(1), reverse=True)[:20]:
    print j,k

# Using Neural Network

import hashlib
import time
from PIL import Image
from operator import itemgetter
import scipy.special
from scipy.misc import imread
import numpy
import os, glob

import dill

from __future__ import division

with open('nn.dill', 'rb') as f: # load the trained Neural Network
    nn = dill.load(f)

char_number_map = {0:'0', 1:'1', 2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'a',11:'b',12:'c',13:'d',14:'e',
                   15:'f',16:'g',17:'h',18:'i',19:'j',20:'k',21:'l',22:'m',23:'n',24:'o',25:'p',26:'q',27:'r',28:'s',
                   29:'t',30:'u',31:'v',32:'w',33:'x',34:'y',35:'z',36:'A',37:'B',38:'C',39:'D',40:'E',41:'F',42:'G',
                   43:'H',44:'I',45:'J',46:'K',47:'L',48:'M',49:'N',50:'O',51:'P',52:'Q',53:'R',54:'S',55:'T',
                  56:'U',57:'V',58:'W',59:'X',60:'Y',61:'Z'}

scores = []
files = os.listdir("./testcaptcha") # go through each of the CAPTCHA in the test set
for file in files:
    if file == ".DS_Store": continue
    im = Image.open("./testcaptcha/" + file)
    im = im.convert("P")
    im2 = Image.new("P", im.size, 255)

    temp = {}

    for x in range(im.size[1]): # convert the image to grayscale
        for y in range(im.size[0]):
            pix = im.getpixel((y,x))
            temp[pix] = pix
            if pix <=52:  
                im2.putpixel((y,x),0)

    im2.save("./testcapt-grayed/" + file.split(".")[0] + ".gif")
    
    inside = False
    letter = False 
    start = 0
    end = 0
    letters = []
    for y in range(im2.size[0]): # segment the characters
        for x in range(im2.size[1]): 
            pix = im2.getpixel((y,x))
            if pix != 255:
                inside = True
        if letter == False and inside == True:
            letter = True
            start = y

        if letter == True and inside == False:
            letter = False
            end = y
            if (end-start) > 5:
                letters.append((start,end))
        inside=False

    count = 0
    size = (60,60) # resize the characters to 60x60 dimension
    captcha = []
    for letter in letters:
        im3 = im2.crop(( letter[0], 0, letter[1], im2.size[1] ))
        im3.thumbnail(size, Image.ANTIALIAS)
        background = Image.new("1", size, 255)
        background.paste(
            im3,
            ((size[0] - im3.size[0]) // 2, (size[1] - im3.size[1]) // 2))
        background.save("./testcapt-grayed/%s.gif" % (str(count)))
        individual_image = imread("./testcapt-grayed/" + str(count) +".gif")
        value = individual_image.flatten()
        inputs = (numpy.asfarray(value) // 255 * 0.99 ) + 0.01
        output = numpy.argmax(nn.predict(inputs)) # feed the pixel values to the neural network
        captcha.append(str(char_number_map[output])) # find the corresponding character for the NN's output using map
        count += 1 # i don't know why i have count here.
    correct = str(file.split(".")[0])
    predicted = ''.join(captcha)
    print "Correct: ", correct, " Predicted: ",  predicted 
    if (correct == predicted):
        scores.append(1)
    else:
        scores.append(0)
        pass
    filelist = glob.glob("./testcapt-grayed/*.gif")
    for f in filelist:
        os.remove(f)
            
scores_array = numpy.asarray(scores)

print "Accuracy = ", scores_array.sum() / scores_array.size * 100, "%"

print "Sum = ", scores_array.sum()

# Using Tesseract

import hashlib
import time
from PIL import Image
from operator import itemgetter
import scipy.special
import numpy
import os, glob
import pytesseract 
import dill

with open('nn.dill', 'rb') as f:
    nn = dill.load(f)

char_number_map = {0:'0', 1:'1', 2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'a',11:'b',12:'c',13:'d',14:'e',
                   15:'f',16:'g',17:'h',18:'i',19:'j',20:'k',21:'l',22:'m',23:'n',24:'o',25:'p',26:'q',27:'r',28:'s',
                   29:'t',30:'u',31:'v',32:'w',33:'x',34:'y',35:'z',36:'A',37:'B',38:'C',39:'D',40:'E',41:'F',42:'G',
                   43:'H',44:'I',45:'J',46:'K',47:'L',48:'M',49:'N',50:'O',51:'P',52:'Q',53:'R',54:'S',55:'T',
                  56:'U',57:'V',58:'W',59:'X',60:'Y',61:'Z'}

scores = []
files = os.listdir("./testcaptcha") 


for file in files:
    if file == ".DS_Store": continue
    im = Image.open("./testcaptcha/" + file)
   
    im = im.convert("P")
    im2 = Image.new("P", im.size, 255)

    temp = {}

    for x in range(im.size[1]):
        for y in range(im.size[0]):
            pix = im.getpixel((y,x))
            temp[pix] = pix
            if pix <=52:  
                im2.putpixel((y,x),0)

    im2.save("./testcapt-grayed/" + file.split(".")[0] + ".gif")
    
    predicted = pytesseract.image_to_string(Image.open('./testcapt-grayed/'+ file.split(".")[0] + ".gif")) 
    
    correct = str(file.split(".")[0])
    print "Correct: ", correct, " Predicted: ",  predicted
    if (correct == predicted):
        scores.append(1)
    else:
        scores.append(0)
        pass
            
scores_array = numpy.asarray(scores)
print "Accuracy = ", scores_array.sum() / scores_array.size * 100, "%"



