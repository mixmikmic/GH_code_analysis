import matplotlib
from libconvolution import convolution
from image_handling import read_image,display_image
from gaussian import gaussian
from debug import time_this_function
get_ipython().magic('matplotlib inline')
import math
import numpy
import setup
setup.compile()

image = read_image('/sample_images/sample_image_1.png')
display_image(image)

g = gaussian()
gaussian_kernel = g.generate_gaussian_smoothening_kernel(0.03)
# gaussian_kernel = [159,[[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]]]
gk = gaussian_kernel[1]
kn = gaussian_kernel[0]

display_image(gk)

# smoothened_image = convolution(image,gaussian_kernel)
smoothened_image = image
display_image(smoothened_image)

sobel_x = [1,[[-1,0,1],[-2,0,2],[-1,0,1]]]
sobel_y = [1,[[-1,-2,-1],[0,0,0],[1,2,1]]]

grad_x = convolution(smoothened_image,sobel_x,flag=False)
grad_y = convolution(smoothened_image,sobel_y,flag=False)

# for row in grad_x : 
#     print(row)
    
print(type(grad_x[0][0]))

magnitude = []
phase = []

def mod(value) : 
    if value < 0 : 
        return -value
    return value

for i in range(len(grad_x)) : 
    magnitude.append([])
    phase.append([])
    for j in range(len(grad_x[i])) :
        magnitude[-1].append(mod(grad_x[i][j]) + mod(grad_y[i][j]))
display_image(magnitude)
# display_image(phase)



