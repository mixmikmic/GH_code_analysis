import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Load the data
train_images = np.load('./data/train_images.npy')
train_labels = np.load('./data/train_labels.npy')
test_images = np.load('./data/test_images.npy')

# Define function to have a look at the data
def show_random_digit( images, labels=None ):
    """"Show a random image out of `images`, 
    with the corresponding label if available"""
    i = np.random.randint(len(images))
    image = images[i].reshape((28, 28))
    plt.imshow( image, cmap='Greys' )
    if labels is not None:
        plt.title('Label: %d' %labels[i])

show_random_digit( train_images, train_labels )

show_random_digit( test_images )

from classification import nearest_neighbor_prediction
get_ipython().magic('pinfo nearest_neighbor_prediction')

get_ipython().run_cell_magic('file', 'serial_script.py', "\nimport numpy as np\nfrom classification import nearest_neighbor_prediction\n\n# Load data\ntrain_images = np.load('./data/train_images.npy')\ntrain_labels = np.load('./data/train_labels.npy')\ntest_images = np.load('./data/test_images.npy')\n\n# Predict the test labels and save it to a file\ntest_labels = nearest_neighbor_prediction( test_images, train_images, train_labels )\nnp.save('data/test_labels_serial.npy', test_labels )")

get_ipython().run_cell_magic('time', '', '! python serial_script.py')

test_labels = np.load('./data/test_labels_serial.npy')

show_random_digit( test_images, test_labels )

