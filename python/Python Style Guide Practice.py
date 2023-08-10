"""
TODO - add a doc string for generate_pi
For an example use the function function_with_types_in_docstring here:
http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
"""
import random

def generate_pi(n):
    """Approximates Pi based on the dart method of throwing darts
    at a unit circle inside of square. For more details see:
    http://interactivepython.org/runestone/static/thinkcspy/Labs/montepi.html

    Args:
        n (int): number of iterations to use in the approximation
        
    Returns:
        float, approximation of Pi
    """
    hit = 0
    for i in range(n):
        if (random.random()**2 + random.random()**2) < 1:
            hit += 1
            
    return hit / float(n) * 4

for i in range(8):
    print('For 10**', i, 'Pi is approximately ', generate_pi(10**i))

"""
TODO - fill in a basic blur function once with an explicit for loop
once with a list comprehension. Check the Google Python Style Guide
for List Comprehension guidelines.
"""

def blur_array_for_loop(to_blur, window, blur_function=sum):
    """Blurs a list using a window size and blur function to apply. For the
    returned list to be the same as the entered list set:
        window = 0 
        blur_function = lambda x: x[0]
        
    For the returned list to be an average of the neighbor values set:
        window = 1 
        blur_function = lambda x: sum(x) / float(len(x))

    Args:
        to_blur (list): list of numerical values
        window (int): number of values to the left and to the right to use
        
    Kwargs:
        blur_function (function): takes in a list and returns a numerical value
        
    Returns:
        list, same length as to_blur where each value is the result of the
            window of values fed into the blur_function.
    """
    # Complete here as a for loop Note this should be implementable inside of 5 lines
    blurred = []
    for i in range(len(to_blur)):
        neighbors = to_blur[max(0, i - window) : min(i + window + 1, len(to_blur))]
        blurred.append(blur_function(neighbors))

    return blurred


def blur_array_list_comprehension(to_blur, window, blur_function=sum):
    """Blurs a list using a window size and blur function to apply. For the
    returned list to be the same as the entered list set:
        window = 0 
        blur_function = lambda x: x[0]
        
    For the returned list to be an average of the neighbor values set:
        window = 1 
        blur_function = lambda x: sum(x) / float(len(x))

    Args:
        to_blur (list): list of numerical values
        window (int): number of values to the left and to the right to use
        
    Kwargs:
        blur_function (function): takes in a list and returns a numerical value
        
    Returns:
        list, same length as to_blur where each value is the result of the
            window of values fed into the blur_function.
    """
    # Complete here as a list comprehensionNote this should be implementable inside of 2 lines
    return [
        blur_function(to_blur[max(0, i - window):min(i + window + 1, len(to_blur))])
        for i in range(len(to_blur))]

    

for_no_blur = blur_array_for_loop(range(100), 0, lambda x: x[0])
for_avg_blur_small = blur_array_for_loop(range(100), 1, lambda x: sum(x) / float(len(x)))
for_avg_blur_large = blur_array_for_loop(range(100), 10, lambda x: sum(x) / float(len(x)))
for_max_blur = blur_array_for_loop(range(100), 10, max)

comp_no_blur = blur_array_list_comprehension(range(100), 0, lambda x: x[0])
comp_avg_blur_small = blur_array_list_comprehension(range(100), 1, lambda x: sum(x) / float(len(x)))
comp_avg_blur_large = blur_array_list_comprehension(range(100), 10, lambda x: sum(x) / float(len(x)))
comp_max_blur = blur_array_list_comprehension(range(100), 10, max)

# Check that the two methods are equivalent
print('Everything should be True:')
print(for_no_blur == comp_no_blur)
print(for_avg_blur_small == comp_avg_blur_small)
print(for_avg_blur_large == comp_avg_blur_large)
print(for_max_blur == comp_max_blur)

"""
TODO - 
"""
import numpy as np

def blur_matrix(to_blur, x_window, y_window, blur_function=sum):
    """Blurs a matrix using a window size and blur function to apply.
    For the returned list to be an average of the neighbor values set:
        window = 1 
        blur_function = lambda x: sum(x) / float(len(x))

    Args:
        to_blur (matrix): matrix of numerical values
        x_window (int): number of values to the left and to the right to use
        y_window (int): number of values to the up and to the down to use
        
    Kwargs:
        blur_function (function): takes in a list and returns a numerical value
        
    Returns:
        list, same length as to_blur where each value is the result of the
            window of values fed into the blur_function.
    """
    # Initialize a zero matrix to store values into
    (x_size, y_size) = to_blur.shape
    blurred = np.zeros((x_size, y_size))
    
    for x in range(x_size):
        for y in range(y_size):
            # TODO Complete here Note this should be implementable inside of 9 lines with only 1 more for loop
            # Find the neighbors of (x, y) in the matrix
            neighbors = []
            for j in range(max(x - x_window, 0), min(x + x_window + 1, x_size - 1)):
                neighbors.extend(to_blur[j][max(y - y_window, 0) : min(y + y_window + 1, y_size)])
            
            blurred[x][y] = blur_function(neighbors)
    return blurred


avg_blur_small = blur_matrix(np.ones((10, 20)), 1, 2, lambda x: sum(x) / float(len(x)))
avg_blur_large = blur_matrix(np.ones((10, 20)), 10, 2, lambda x: sum(x) / float(len(x)))
max_blur = blur_matrix(np.ones((10, 20)), 2, 2, max)

