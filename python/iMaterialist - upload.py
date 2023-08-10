# Import numpy for the high performance in-memory matrix/array storage and operations.
import numpy as np

# Import h5py for the HD5 filesystem high performance file storage of big data.
import h5py

# Import time to record timing
import time

def feeder(files):
    """ A generator for feeding batches of images to a neural network 
    files = batches of images/labels as HD5 files
    """
    # We use an infinite loop here so that the generator can be called for an unlimited number of epochs
    while True:
        # Read each batch file one at a time in sequential order
        for batch_file in files:
            # Read the batch to disk from the HD5 file
            with h5py.File(batch_file, 'r') as hf:
                # Read in the images
                # Normalize the pixel data (convert 0..255 to 0..1)
                X = hf['images'][:] / 255 
                # Read in the corresponding labels
                Y = hf['labels'][:]
                
                # Generator - return list of X (images) and Y (labels)
                yield X, Y

# Install pympler, used for memory monitoring
get_ipython().system('pip install pympler')

import os

# Import the garbage (memory management) module
import gc

# Import pympler.tracker for memory monitoring
from pympler import tracker
# Create object to monitor the heap
tr = tracker.SummaryTracker()

# Directory where the training batches are stored
batches = "contents/train/"

# Get a list of all the training\ batch files
batchlist = []
for batch in os.listdir(batches):
    batchlist.append(os.path.join(batches, batch))
    
# Number of batches
nbatches = len(batchlist)

# Number of epochs
n_epochs = 1
    
# Loop through each epoch, each time feeding the entire training set.
for epoch in range(n_epochs):
    # Iteratively call the feeder() function
    nbatch = 0
    for X, Y in feeder(batchlist):
        # Printing some information so you can see that the next batch file was feed
        print("BATCH #:", nbatch)
        print("X", X.shape)
        print("Y", Y.shape)
        tr.print_diff()
        
        # HERE is where you feed the training batch data to the neural network
        
        # This line will force garbage collection of unused memory.
        gc.collect()
        
        # Run a single epoch
        nbatch += 1
        if nbatch == nbatches:
            break
    



