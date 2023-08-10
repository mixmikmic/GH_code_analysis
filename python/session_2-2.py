import numpy as np
import matplotlib.pyplot as plt

def logistic_function(z):
    return 1./(1+np.exp(-z))

zs = np.linspace(-10,10,1000)


fig = plt.figure(figsize = (8,6))

plt.plot(zs, logistic_function(zs), lw = 3)
plt.plot([-10,10],[0.5,0.5], ls = '--', c = 'k')

plt.xlim(-10,10)
plt.ylim(-.1, 1.1)

plt.title('Logistic function', size = 18)
plt.xlabel('z', size = 16)
plt.ylabel('f(z)', size = 16)
plt.grid()
plt.tick_params(labelsize = 16)

plt.show()

##########################
# Insert solution below! #
##########################





##########################
# Insert solution below! #
##########################




















##########################
# Insert solution below! #
##########################
import time


# Put your code between the two time.time() functions if you wish to benchmark the fitting time
t0 = time.time()









t1 = time.time()
print('time elapsed (s):\t\t', t1-t0)

##########################
# Insert solution below! #
##########################






##########################
# Insert solution below! #
##########################







##########################
# Insert solution below! #
##########################




# Here's a freebie :D
false_positive_rate = len(np.where(y_test[y_test == 0] != y_test_predictions[y_test == 0])[0])/N_test







##########################
# Insert solution below! #
##########################








##########################
# Insert solution below! #
##########################







##########################
# Insert solution below! #
##########################








##########################
# Insert solution below! #
##########################





# Put your code between the two time.time() functions if you wish to benchmark the fitting time
t0 = time.time()








t1 = time.time()
print('time elapsed (s):\t\t', t1-t0)

import scipy

def PreprocessImage(image_file_path):
    # Set up fig
    fig, axes = plt.subplots(3, 3, figsize = (12,12))
    
    custom_digit = plt.imread(image_file_path)[:,:,[0,1,2]]
    
    # Load in .png image
    plt.sca(axes[0,0])
    plt.imshow(custom_digit)
    plt.title('raw image')
    


    # Remove alpha channel
    custom_digit = custom_digit[:,:,[0,1,2]]

    


    # Rotate image
    # This step may not be necessary; my phone always rotates images for some reason...
    plt.sca(axes[0,1])
    
    custom_digit = np.rot90(custom_digit, axes = (1,0))
    plt.imshow(custom_digit, cmap = 'gray')
    plt.title('rotated image')


    # Average RGB channels
    plt.sca(axes[0,2])
    
    custom_digit = np.mean(custom_digit, axis = 2)
    plt.imshow(custom_digit, cmap = 'gray')
    plt.title('convert to grayscale')
    


    # Invert grays
    plt.sca(axes[1,0])
    
    custom_digit = 255-custom_digit
    plt.imshow(custom_digit, cmap = 'gray')
    plt.title('invert grays')
    


    # Histogram
    plt.sca(axes[1,1])
    
    plt.hist(custom_digit.flatten(), bins = 255)
    plt.yscale('log', nonposy='clip')
    
    plt.title('intensity histogram')
    


    # Threshold background to black
    
    # Set the correct threshold to separate the dark background from the light foreground (the letters)
    # The threshold should be between the very large background peak and the smaller peak in the middle
    
    plt.sca(axes[1,2])

    threshold = 175
    custom_digit[custom_digit < threshold] = 0
    plt.imshow(custom_digit, cmap = 'gray')
    
    plt.title('threshold background to black')
    
    # Rescale
    plt.sca(axes[2,0])
    
    custom_digit = custom_digit*255./np.max(custom_digit)
    
    plt.imshow(custom_digit, cmap = 'gray')
    
    plt.title('rescale brightness')


    # Crop image
    plt.sca(axes[2,1])
    
    
    threshold = 1
    digit_top_row = np.where(np.mean(custom_digit, axis = 1) > threshold)[0][0]
    digit_bottom_row = np.where(np.mean(custom_digit, axis = 1) > threshold)[0][-1]
    digit_left_column = np.where(np.mean(custom_digit, axis = 0) > threshold)[0][0]
    digit_right_column = np.where(np.mean(custom_digit, axis = 0) > threshold)[0][-1]

    center = [int((digit_bottom_row + digit_top_row)/2.), int((digit_right_column + digit_left_column)/2.)]

    width = int(2.5*(digit_right_column - digit_left_column))

    new_top_row = center[0] - int(width/2.)
    new_bottom_row = new_top_row + width

    new_left_column = center[1] - int(width/2.)
    new_right_column = new_left_column + width


    custom_digit = np.copy(custom_digit[new_top_row:new_bottom_row + 1, new_left_column:new_right_column + 1])
    plt.imshow(custom_digit, cmap = 'gray')
    
    plt.title('crop')
    
    print('debugging info')
    print(digit_top_row, digit_bottom_row, digit_left_column, digit_right_column, center, width)
    

    # Down sample image
    plt.sca(axes[2,2])
    custom_digit = scipy.misc.imresize(custom_digit, size = (28, 28))
    plt.imshow(custom_digit, cmap = 'gray')
    
    plt.title('down sample')
    
    
    plt.show()
    
    return custom_digit.flatten()


# ~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~
# Replace this string with the location of your own file!
# ~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~

custom_digit = PreprocessImage('./data/digits/hand-drawn_7_0.JPG')

print(model.predict)

# Load in .png image
custom_digit = plt.imread('../data/digits/custom_digit.png')[:,:,[0,1,2]]

# Remove alpha channel
custom_digit = custom_digit[:,:,[0,1,2]]

# Average RGB channels
custom_digit = np.mean(custom_digit, axis = 2)

# Rescale [0-1]->[0,255]
custom_digit = custom_digit*255

plt.imshow(custom_digit, cmap = 'gray', vmin = 0, vmax = 255, interpolation = 'none')
plt.show()

print(model.predict(custom_digit.flatten()))

