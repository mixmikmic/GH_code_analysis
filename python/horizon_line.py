from pathlib import Path
import cv2
import numpy as np
import math

# Paths to images
root_dir = Path.cwd()
images_path = root_dir / '..' / 'test_images'
output_path = root_dir / '..' / 'output_images'

def custom_canny(img, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img, lower, upper)
    return edged


def plot_line(img, rho, theta):
    # Plots the line coming out of a Hough Line Transform
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * (a)))
    pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * (a)))
    cv2.line(img, pt1, pt2, (255, 0, 0), 3)

for image_path in list(images_path.glob('*.png')):
    image = cv2.imread(str(image_path))
    # Blur image and convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Use Canny edge detection to find edges
    edges = custom_canny(blurred)
    # Dilate edges of lines
    dilated = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8))
    # Use Hough Line Transform to get lines
    lines = cv2.HoughLines(dilated, 1, np.pi / 100,
                           threshold = 400,
                           min_theta=np.pi / 3,
                           max_theta=2 * np.pi / 3)
    if lines is not None:
        print('Found %s lines' % (len(lines)))
        # # Print all lines
        # for line in lines:
        #     for rho, theta in line:
        #         plot_line(edges, rho, theta)
        # Average line only
        avg_rho = np.mean([line[0][0] for line in lines])
        avg_theta = np.mean([line[0][1] for line in lines])
        plot_line(image, avg_rho, avg_theta)
    else:
        print('No Horizon Found')

    # Output images to window
    # cv2.imshow("Original", image)
    # cv2.imshow("Edges", edges)
    # cv2.waitKey(0)
    
    # Save images to file
    cv2.imwrite(str(output_path / image_path.name), image)

