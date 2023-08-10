from SimpleCV import Camera
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

# Initialize the camera
cam = Camera()
# Loop to continuously get images

# Get Image from camera
img = cam.getImage()

img = img.binarize()
    # Draw the text "Hello World" on image
img.drawText("Hello World!")
    # Show the image
#img.show()

type(img)

plt.imshow(img)   



