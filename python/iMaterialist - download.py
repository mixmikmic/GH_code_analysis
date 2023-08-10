# Get the latest version of pip
get_ipython().system('python -m pip install --upgrade pip')

# Install the Kaggle module
get_ipython().system('pip install kaggle')

get_ipython().system('kaggle competitions download -c imaterialist-challenge-furniture-2018')

# numpy for the high performance in-memory matrix/array storage and operations.
get_ipython().system('pip install numpy   ')
# h5py for the HD5 filesystem high performance file storage of big data.
get_ipython().system('pip install h5py   ')
# Python image manipulation library (replaces PIL)
get_ipython().system('pip install pillow  ')
# requests for HTTP operations
get_ipython().system('pip install requests')

# Import numpy for the high performance in-memory matrix/array storage and operations.
import numpy as np

# Import h5py for the HD5 filesystem high performance file storage of big data.
import h5py

# Import PIL.Image for Python image manipulation library. 
from PIL import Image

# Import json and requests for HTTP operations
import json, requests

# Import the Byte and String IO library for extracing data returned (response) frome HTTP requests.
from io import BytesIO, StringIO

# Import time to record timing
import time

# Library for thread execution
import threading

def load_dispatcher(url, batch_type, batch_size=200, size=(300,300), grayscale=False, normalize=False, concurrent=5):
    """ Load the Data in Batches 
    url - location of data dictionary
    batch_type - training, validation or test
    batch_size - size of the batch
    size - size to rescale image to
    grayscale - flag to convert image to grayscale
    concurrent - the number of concurrent (parallel)) batch loads
    """
    
    # First retreive the dataset dictionary, which is in a JSON format. 
    # Dictionary is stored remote: We will make a HTTP request
    if url.startswith("http"):
        datadict = json.load( requests.get(url).content )
    # Dictionary is stored locally
    else:
        datadict = json.load( open( url ) )
   
    # The number of batches
    batches = int(len(datadict['images']) / batch_size)
    
    # Sequentially Load each batch group (i.e., concurrent)
    for i in range(0, batches, concurrent):
        # Start time for the batch group
        start_time = time.time()
        
        # Listof threads, corresponding to to the processing of each batch in the batch group
        threads = []
        # Create and Start a processing thread for each batch in the batch group
        for j in range(concurrent):
            t = threading.Thread(target=load_and_store_batch, args=(datadict, batch_type, i + j, batch_size, size, grayscale, normalize, ))
            # Keep track (remember) of the thread
            threads.append(t)
            # Start the thread
            t.start()
        # Join the threads into a single wait for all threads to complete
        for t in threads:
            t.join()
                  
        # Calculate elapsed time in seconds to load this batch group
        elapse = int(time.time() - start_time)
            
        # Estimate remaining time in minutes for loading remaining barches.
        remaining = int( ( ( batches - i ) / concurrent ) * elapse ) / 60
        
        print("Remaining time %d mins" % remaining)

def load_and_store_batch(datadict, batch_type, pos, batch_size, size, grayscale, normalize):
    """ Process loading (extration), handling (transformation) and storing (loading) as a batch 
    batch_type - training, validation or test
    pos - the batch slice position in the data (i.e., the first, the second, etc)
    batch_size - size of the batch
    size - size to rescale image to
    grayscale - flag to convert image to grayscale
    """
    start_time = time.time()
    
    start = pos * batch_size
    images, labels = load_batch(datadict, start, batch_size, size, grayscale, normalize )
        
    # Calculate elapsed time in seconds to load this batch
    elapse = int(time.time() - start_time)
        
    print("Batch Loaded %d: %d secs" % (pos, elapse))
        
    # Write the batch to disk as HD5 file
    with h5py.File('contents\\' + batch_type + '\\images' + str(pos) + '.h5', 'w') as hf:
        hf.create_dataset("images",  data=images)
    #with h5py.File('contents\\' + batch_type + '\\labels' + str(pos) +  '.h5', 'w') as hf:
        hf.create_dataset("labels",  data=labels)

timeout = 6   # timeout (seconds) for reading the image from the web
retries = 2   # Number of times to retry reading the image over the network

def load_batch(datadict, start, batch_size, size, grayscale, normalize):
    """ Load the training datas
    datadict - data image/label dictionary
    start - index to start reading batch of images
    batch_size - number of images to read (None = all images)
    grayscale - flag if image should be converted to grayscale
    """
    
    images = [] # List containing the images
    labels = [] # List containing the corresponding labels for the images
    
    # Number of images to load
    if batch_size == None:
        batch_size = len(datadict['images'])
      
    # Final shape of image Height, Width
    if grayscale == True:
        shape = size
    # Final shape of image Height, Width, Channels(3)
    else:
        shape = size + (3,)
        
    not_loaded = 0 # Number of images that failed to load in the batch
            
    # Load the batch of images/labels from the Data Dictionary
    end = start + batch_size
    for i in range(start, end): 
        image_url = datadict['images'][i]['url'][0]
        label_id  = datadict['annotations'][i]['label_id']

        # Keep trying to read the image over the network on failure upto retries number of times
        for retry in range(retries):
            # Download, resize and convert images to arrays
            try:
                # Make HTTP request fot the image data
                response = requests.get(image_url, timeout=10)

                # Use the PIL.Image libary to load the image data as au uncompressed RGB or Grayscale bitmap
                if grayscale == True:
                    pixels = Image.open(BytesIO(response.content)).convert('LA')
                else:
                    pixels = Image.open(BytesIO(response.content))

                # Resize the image to be all the same size
                pixels = pixels.resize(size, resample=Image.LANCZOS)

                # Load the image into a 3D numpy array
                image = np.asarray(pixels)

                # Discard image if it does not fit the final shape
                if image.shape != shape:
                    if grayscale == False:
                        # Was a gray scale image
                        if image.shape == size:
                            # Extend to three channels, replicating the single channel
                            pixels = pixels.convert('RGB')
                            image = np.asarray(pixels)
                            break
                        # Is RGBA image (4 channels)
                        if image.shape == size + (4,):
                            # Remove Alpha Channel from Image
                            pixels = pixels.convert('RGB')
                            image = np.asarray(pixels)
                            break
                            
                    # Unrecognized shape
                    not_loaded += 1
                    retry = retries
                    break
            except Exception as ex:
                if retry < retries-1:
                    continue
                #print("CAN'T FETCH IMAGE", image_url)
                retry = retries
            # Image was read or failed retries number of times
            break
                
        if retry == retries:
            not_loaded += 1
            continue

        # if bad image, skip
        if np.any(image == None):
            continue
            
        # Normalize the image (convert pixel values from int range 0 .. 255 to float range 0 .. 1)
        if normalize == True:
            image = image / 255
            
        # add image to images list
        images.append( image )
        # add corresponding label to labels list
        labels.append( label_id )
        
        if (i+1) % 50 == 0:
            print('%d Images added, %d not loaded' % ((i + 1), not_loaded))

    return images, labels
        

# Create Directories for the HD5 encoded batches
get_ipython().system('mkdir contents')
get_ipython().system('mkdir contents\\\\train')
get_ipython().system('mkdir contents\\\\validation')
get_ipython().system('mkdir contents\\\\test')

# Data dictionaries
train_url      = 'C:\\Users\\User\\.kaggle\\competitions\\imaterialist-challenge-furniture-2018\\train.json'
test_url       = 'C:\\Users\\User\\.kaggle\\competitions\\imaterialist-challenge-furniture-2018\\test.json'
validation_url = 'C:\\Users\\User\\.kaggle\\competitions\\imaterialist-challenge-furniture-2018\\validation.json'

# Load the Training Batches
load_dispatcher(train_url, "train")

# Load the Validation Batches
load_dispatcher(validation_url, "validation")

# Load the Test Batches
load_dispatcher(test_url, "test")





