get_ipython().magic('matplotlib inline')
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from PIL import Image
from skimage import data
import scipy
import math
from scipy.ndimage import measurements
from skimage import data
from ipywidgets import interact, fixed, FloatSlider, IntSlider,FloatRangeSlider, Label

def display_line_result(og_img, hough_img, edges):
    
    current_image=data.checkerboard()
    w, h = current_image.shape
    output_image = np.empty((w, h, 3))
    edges = cv2.Canny(current_image,50,150,apertureSize =3)
    output_image[:, :, 2] =  output_image[:, :, 1] =  output_image[:, :, 0] =  current_image/255.
    lines = cv2.HoughLines(edges,1,np.pi/180,120)
    max_size=max(w,h)**2
    for rho_theta in lines:
        rho=rho_theta[0][0]
        theta=rho_theta[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + max_size*(-b))
        y1 = int(y0 + max_size*(a))
        x2 = int(x0 - max_size*(-b))
        y2 = int(y0 - max_size*(a))
        cv2.line(output_image,(x1,y1),(x2,y2),(1,0,0),1)
    
    
    fig2, axes_array = plt.subplots(1, 4)
    fig2.set_size_inches(9,3)
    image_plot = axes_array[0].imshow(og_img,cmap=plt.cm.gray) 
    axes_array[0].axis('off')
    axes_array[0].set(title='Original Image')
    image_plot = axes_array[1].imshow(edges,cmap=plt.cm.gray)
    axes_array[1].axis('off')
    axes_array[1].set(title='Edged Image')
    image_plot = axes_array[2].imshow(hough_img)
    axes_array[2].axis('off')
    axes_array[2].set(title='Hough Lines Image')
    image_plot = axes_array[3].imshow(output_image)
    axes_array[3].axis('off')
    axes_array[3].set(title='Hough Lines Open CV')
    plt.show()
    return

def get_canny(og_img):
    edged_image = cv2.Canny(og_img,50,150,apertureSize = 3)#current_image=data.checkerboard()
    return edged_image

def hough_lines(og_img,rho_resolution,theta_resolution,threshold):
    rho_theta_values = []
    width, height = og_img.shape
    hough_img = np.empty((width, height, 3))
    hough_img[:, :, 2] =  hough_img[:, :, 1] =  hough_img[:, :, 0] =  og_img/255.
    
    digonal = math.sqrt(width*width + height*height)
    max_size=max(width,height)**2
    
    thetas = np.linspace(0,180,theta_resolution+1)
    rhos = np.linspace(-digonal,digonal,rho_resolution+1)
   
    acc = np.zeros((rho_resolution+1,theta_resolution+1))

    for x_index in range(0, width):
        for y_index in range(0, height):
            if edges[x_index][y_index] > 0:
                for t_index in range(0, len(thetas)):
                    rho = x_index * math.cos(thetas[t_index]) + y_index * math.sin(thetas[t_index])
                    for r_index in range(0, len(rhos)):
                        if rhos[r_index]>rho:
                            break
                    acc[r_index][t_index] += 1
   
    for rho_value in range(0, len(rhos)):
        for t_value in range(0, len(thetas)):
            if acc[rho_value][t_value] >= threshold:
                rho_theta_values.append([rhos[rho_value], thetas[t_value]])

    
    for rho_theta in rho_theta_values:
        rho=rho_theta[0]
        theta=rho_theta[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + max_size*(-b))
        y1 = int(y0 + max_size*(a))
        x2 = int(x0 - max_size*(-b))
        y2 = int(y0 - max_size*(a))
        cv2.line(hough_img,(x1,y1),(x2,y2),(1,0,0),1)
    
    display_line_result(og_img, hough_img, edges)
    return
    
og_img = data.checkerboard()
edges = get_canny(og_img)

interact(hough_lines,
         og_img = fixed(og_img),
         rho_resolution=IntSlider(min=10, max=1000, step=1,value=150,continuous_update=False),
         theta_resolution=IntSlider(min=10, max=1000, step=1,value=360,continuous_update=False),
         threshold=IntSlider(min=5, max=1000, step=1,value=180,continuous_update=False)) 

