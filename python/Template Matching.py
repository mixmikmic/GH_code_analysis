# manipulation
import numpy as np
import cv2

# displaying images
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')

# edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# read the image
frame = (mpimg.imread('frame.png')*255).astype(np.uint8)

# display image properties and image
print('This image is:', frame.dtype, type(frame), 'with dimensions:', frame.shape)
plt.imshow(frame) 

# convert image to grayscale
grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
plt.imshow(grayFrame, cmap='gray')

# Template is just a part of the original frame
mario = grayFrame[178:208,112:128]
plt.imshow(mario, cmap='gray')

# find the template
ssd = cv2.matchTemplate(grayFrame, mario, method=cv2.TM_SQDIFF) # sum of square difference image
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(ssd) # we care about the minLoc

# display the template
mh, mw = mario.shape # for the height and width of the box we're drawing
x, y = minLoc # minLoc is a point (x, y)
cv2.rectangle(frame, (x, y), (x+mw, y+mh), color=[255,255,0], thickness=1) # draw rectangle
plt.imshow(frame)

def trackTemplate(frame, templ):
    
    # convert frame to grayscale
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # match template
    ssd = cv2.matchTemplate(grayFrame, templ, method=cv2.TM_SQDIFF)
    minLoc = cv2.minMaxLoc(ssd)[2]

    # draw box around template location
    th, tw = templ.shape[:2] # for the height and width of the box we're drawing
    x, y = minLoc # minLoc is a point (x, y)
    cv2.rectangle(frame, (x, y), (x+tw, y+th), color=[255,255,0], thickness=1) # draw rectangle
    
    return frame

outFile1 = "output1.mp4"
vid = VideoFileClip("smb.mp4")

# this function processes the frames. 
# Note: fl_image takes in a one parameter function as an argument and passes the frame to it.
#       To send a function with multiple arguments, create an inline anonymous function.
outVid1 = vid.fl_image(lambda frame: trackTemplate(frame, mario))

# write to file
get_ipython().magic('time outVid1.write_videofile(outFile1, audio=False)')

HTML("""
<video src="{0}" controls />
""".format(outFile1))

mario = mario[1:12,2:]
mask = np.array([
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
])
plt.imshow(mario*mask, cmap='gray')

def trackTemplate(frame, templ, mask):
    
    # convert frame to grayscale
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # match template
    ssd = cv2.matchTemplate(grayFrame, templ, method=cv2.TM_SQDIFF, mask=mask)
    minLoc = cv2.minMaxLoc(ssd)[2]

    # draw box around template location
    th, tw = templ.shape[:2] # for the height and width of the box we're drawing
    x, y = minLoc # minLoc is a point (x, y)
    cv2.rectangle(frame, (x-1, y), (x+14, y+30), color=[255,255,0], thickness=1) # draw rectangle
    
    return frame

outFile2 = "output2.mp4"
vid = VideoFileClip("smb.mp4")
outVid2 = vid.fl_image(lambda frame: trackTemplate(frame, mario, mask))
get_ipython().magic('time outVid2.write_videofile(outFile2, audio=False)')

HTML("""
<video src="{0}" controls />
""".format(outFile2))

HTML("""
<iframe width="560" height="315" 
    src="https://www.youtube.com/embed/m99H6oH46E8?ecver=1" 
    frameborder="0" allowfullscreen>
</iframe>""")



