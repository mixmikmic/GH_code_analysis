import numpy as np
import cv2

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def detect_edges_canny(image_path, th1 = 100, th2 = 200):

    # Load image
    img = cv2.imread(image_path)
    
    # Detect edges
    edges = cv2.Canny(img, threshold1= th1, threshold2 = th2)

    # Plot original and edge image
    plt.figure(figsize=(10, 8))
    plt.subplot(121); plt.imshow(img, cmap='gray')
    plt.title('Original Image'); plt.axis('off');
    plt.subplot(122); plt.imshow(edges, cmap='gray')
    plt.title('Image with Edges'); plt.axis('off')
    plt.show();

detect_edges_canny('images/german_street.jpg', th1 = 100, th2 = 200)

detect_edges_canny('images/german_street.jpg', th1= 200, th2 = 250)

detect_edges_canny('images/road_scene.jpg', th1 = 100, th2 = 200)

detect_edges_canny('images/road_scene.jpg', th1 = 150, th2 = 250)

def detect_edges_sobel(image_path):

    # Load image
    img = cv2.imread(image_path)
    
    img_binary = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    edgeX = cv2.Sobel(img_binary, cv2.CV_16S, 0, 1)
    edgeY = cv2.Sobel(img_binary, cv2.CV_16S, 1, 0)
    
    
    # Detect edges
    edgeX = np.uint8(np.absolute(edgeX))
    edgeY = np.uint8(np.absolute(edgeY))
    edge = cv2.bitwise_or(edgeX, edgeY)
    
    plt.figure(figsize=(10, 8))
    plt.subplot(121); plt.imshow(edgeX, cmap='gray')
    plt.title('X Edges'); plt.axis('off');
    plt.subplot(122); plt.imshow(edgeY, cmap='gray')
    plt.title('Y Edges'); plt.axis('off')

    # Plot original and edge image
    plt.figure(figsize=(10, 8))
    plt.subplot(121); plt.imshow(img)
    plt.title('Original Image'); plt.axis('off');
    plt.subplot(122); plt.imshow(edge, cmap='gray')
    plt.title('Image with Edges'); plt.axis('off')
    plt.show();

detect_edges_sobel('images/german_street.jpg')

detect_edges_sobel('images/road_scene.jpg')



