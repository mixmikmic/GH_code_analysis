# Import packages
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract
import argparse
import cv2
import os

#Construct and Parse The Argument 
#parser = argparse.ArgumentParser()
#parser.add_argument("-i", "--image", required = True, help = "Path to the image")
#parser.add_argument("-p", "--preprocess", type=str, default="thresh", help="type of preprocessing to be done")

#args = vars(parser.parse_args())

# Load an color image in grayscale 
#img = cv2.imread(args["image"],0) 

img = cv2.imread("sample.png",0) 

#Write the grayscale image to disk as a temporary file
filename = "{}.png".format(os.getpid())
print(filename)

cv2.imwrite(filename, img)

# Load the image using PIL (Python Imaging Library), Apply OCR, and then delete the temporary file
text = pytesseract.image_to_string(Image.open(filename))
print(text)
os.remove(filename)

# Using OpenCV
#cv2.imshow("Image", img)
#cv2.waitKey(0) ## cv2.waitKey() The function waits for specified milliseconds for any keyboard event. 0 means, it waits indefinitely for a key stroke.

#OR

# Using Matplotlib
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()



