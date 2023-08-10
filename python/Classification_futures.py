import numpy as np
import matplotlib.pyplot as plt
from classification import nearest_neighbor_prediction
get_ipython().magic('matplotlib inline')

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Define function
def is_even( i ):
    """Determine if the integer i is even"""
    even = (i%2==0)
    return( even )

list_integers = [ 4, 3, 6, 7, 9, 10, 124, 325 ]

# Spawn 4 processes
with ProcessPoolExecutor(max_workers=4) as e:
    result = e.map( is_even, list_integers )
    
for answer in result:
    print(answer)

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

def predict( test_images ):
    return( nearest_neighbor_prediction( test_images, train_images, train_labels ) )

get_ipython().run_cell_magic('time', '', 'test_labels_serial = predict( test_images )')

# Choose the number of processes and split the data
N_processes = 4
split_arrays = np.array_split( test_images, N_processes )

get_ipython().run_cell_magic('time', '', 'with ProcessPoolExecutor(max_workers=N_processes) as e:\n    result = e.map( predict, split_arrays )\n\n# Merge the result from each process into a single array\ntest_labels_proc = np.hstack( ( small_test_labels for small_test_labels in result ) )')

show_random_digit( test_images, test_labels_proc )

# Choose the number of threads and split the data
N_threads = 4
split_arrays = np.array_split( test_images, N_threads )

get_ipython().run_cell_magic('time', '', 'with ThreadPoolExecutor(max_workers=N_threads) as e:\n    result = e.map( predict, split_arrays )\n    \n# Merge the result from each thread into a single array\ntest_labels_threads = np.hstack( ( small_test_labels for small_test_labels in result ) )')

show_random_digit( test_images, test_labels_threads )

