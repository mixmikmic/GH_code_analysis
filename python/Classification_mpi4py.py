import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

get_ipython().run_cell_magic('file', 'example.py', "\nfrom mpi4py.MPI import COMM_WORLD as communicator\nimport random\n\n# Draw one random integer between 0 and 100\ni = random.randint(0, 100)\nprint('Rank %d' %communicator.rank + ' drew a random integer: %d' %i )\n\n# Gather the results\ninteger_list = communicator.gather( i, root=0 )\nif communicator.rank == 0:\n    print('\\nRank 0 gathered the results:')\n    print(integer_list)")

get_ipython().system(' mpirun -np 3 python example.py')

get_ipython().run_cell_magic('file', 'parallel_script.py', "\nfrom classification import nearest_neighbor_prediction\nimport numpy as np\nfrom mpi4py.MPI import COMM_WORLD as communicator\n\n# Load data\ntrain_images = np.load('./data/train_images.npy')\ntrain_labels = np.load('./data/train_labels.npy')\ntest_images = np.load('./data/test_images.npy')\n\n# Use only the data that this rank needs\nN_test = len(test_images)\nif communicator.rank == 0:\n    i_start = 0\n    i_end = N_test/2\nelif communicator.rank == 1:\n    i_start = N_test/2\n    i_end = N_test    \nsmall_test_images = test_images[i_start:i_end]\n\n# Predict the results\nsmall_test_labels = nearest_neighbor_prediction(small_test_images, train_images, train_labels)\n\n# Assignement: gather the labels on one process and have it write it to a file\n# Hint: you can use np.hstack to merge a list of arrays into a single array, \n# and np.save to save an array to a file.")

get_ipython().run_cell_magic('time', '', '! mpirun -np 2 python parallel_script.py')

# Load and split the set of test images
test_images = np.load('data/test_images.npy')
split_arrays_list = np.array_split( test_images, 4 )

# Print the corresponding shape
print( 'Shape of the original array:' )
print( test_images.shape )
print('Shape of the splitted arrays:')
for array in split_arrays_list:
    print( array.shape )

get_ipython().run_cell_magic('file', 'parallel_script.py', "\nfrom classification import nearest_neighbor_prediction\nimport numpy as np\nfrom mpi4py.MPI import COMM_WORLD as communicator\n\n# Load data\ntrain_images = np.load('./data/train_images.npy')\ntrain_labels = np.load('./data/train_labels.npy')\ntest_images = np.load('./data/test_images.npy')\n\n# Assignement: use the function np.array_split the data `test_images` among the processes\n# Have each process select their own small array.\nsmall_test_images = #.....\n\n# Predict the results and gather it on rank 0\nsmall_test_labels = nearest_neighbor_prediction(small_test_images, train_images, train_labels)\n\n# Assignement: gather the labels on one process and have it write it to a file\n# Hint: you can use np.hstack to merge a list of arrays into a single array, \n# and np.save to save an array to a file.")

get_ipython().run_cell_magic('time', '', '! mpirun -np 4 python parallel_script.py')

# Load the data from the file
test_images = np.load('data/test_images.npy')
test_labels_parallel = np.load('data/test_labels_parallel.npy')

# Define function to have a look at the data
def show_random_digit( images, labels=None ):
    """"Show a random image out of `images`, 
    with the corresponding label if available"""
    i = np.random.randint(len(images))
    image = images[i].reshape((28, 28))
    plt.imshow( image, cmap='Greys' )
    if labels is not None:
        plt.title('Label: %d' %labels[i])

show_random_digit( test_images, test_labels_parallel )

