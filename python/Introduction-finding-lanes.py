import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
get_ipython().magic('pylab inline')

# Read in the image and print out some stats
image = mpimg.imread('test.jpg')
print('This image is: ', type(image),
     'with dimensions:', image.shape)

plt.imshow(image)
plt.title('Original Image')

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]

print('xsize: ', xsize,
     '\nysize: ', ysize)

color_select = np.copy(image)

# Define our color selection criteria
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Identify pixels below the threshold 
color_thresholds = (image[:,:,0] < rgb_threshold[0])             | (image[:,:,0] < rgb_threshold[0])             |(image[:,:,0] < rgb_threshold[0])
        
print('Thresholds\n\n', color_thresholds)

# Mask color selection 
color_select[color_thresholds] = [0, 0, 0]

# Display the color_select image
f,(ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,10))
ax1.imshow(image)
ax2.imshow(color_select)
ax1.set_title('Original image')
ax2.set_title('Color Selected image')

plt.imshow(color_select)

# make a copy of the image
region_select = np.copy(image)

# Define a triangle region of interest
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing

left_bottom = [130, 539]
right_bottom = [900, 539]
apex = [465, 320]

# Fit lines (y=Ax+B) to identify the 3 sided region of interest
# np.polyfit() returns the coefficients [A, B] of the fit

fit_left = np.polyfit((left_bottom[0], apex[0]),
                      (left_bottom[1], apex[1]),1)
                     
fit_right = np.polyfit((right_bottom[0], apex[0]),
                      (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]),
                       (left_bottom[1], right_bottom[1]), 1)

print('fit_left: ', fit_left)
print('fit_right: ', fit_right)
print('fit_bottom: ', fit_bottom)

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
print(XX)
print(YY)

region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) &                     (YY > (XX*fit_right[0] + fit_right[1])) &                     (YY < (XX*fit_bottom[0] + fit_bottom[1]))
        
print(region_thresholds)

# Mask color and region selection
region_select[region_thresholds] = [250, 0, 0]

plt.imshow(region_select)
plt.title('Region of Interest')

line_image = np.copy(image)

# Find where image is both colored right and in the region 
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

plt.imshow(line_image)

# Color pixels red where both color and region selections meet
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

# Display the image
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(20,15))
ax1.imshow(image)
ax2.imshow(color_select)
ax3.imshow(region_select)
ax4.imshow(line_image)
ax1.set_title('Original Image')
ax2.set_title('Color Selection Image')
ax3.set_title('Region Selection Image')
ax4.set_title('Combined Color and Region Selection Image')
plt.show()

image = mpimg.imread('exit-ramp.jpg')
plt.imshow(image)

# Import OpenCV
import cv2

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')

# Define parameters for Canny
low_threshold = 100
high_threshold = 200
edges = cv2.Canny(gray, low_threshold, high_threshold)

# Display the result
plt.imshow(edges, cmap='Greys_r')

image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image)

plt.imshow(gray, cmap='gray')

# Define parameters for Canny
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(gray, low_threshold, high_threshold)

# Create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)
ignore_mask_color = 255

# Define a four side polygon region to mask
imshape = image.shape
vertices = np.array([[(0, imshape[0]), (450, 290), (490, 290), 
                      (imshape[1], imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

help(cv2.HoughLinesP)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on

rho = 2
theta = np.pi/180
threshold = 20
min_line_length = 15
max_line_gap = 20

line_image = np.copy(image)*0 # creating a blank to draw lines on 

# Run Hough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), 
                       min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on the blank
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (250, 0, 0), 10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges))

# Draw the lines on the edge image
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
plt.imshow(lines_edges)

