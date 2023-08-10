import matplotlib.pyplot as plt
import numpy as np

def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i
    return max_iters

def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height, width = image.shape
    
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color

get_ipython().run_cell_magic('time', '', '\nimage = np.empty((1024, 1536), dtype = np.uint8)\ncreate_fractal(-2.0, 1.0, -1.0, 1.0, image, 20) \n\nplt.imshow(image)\nplt.show()')



