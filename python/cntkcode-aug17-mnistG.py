#mnistG.py - graph API usage for a conv net with cntk
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import cntk as C
import datetime

print("CNTK version",C.__version__)
#let's set up safety and tracing flags
C.debugging.set_computation_network_trace_level(1) #1000 is more and 1000000 is intense

print("CNTK sees device(s): ",C.all_devices())
#Ensure we always get the same randomizer init
np.random.seed(0)

# Define the data dimensions
input_dim_model = (1, 28, 28)    # images are 28 x 28 with 1 channel of color (gray)
input_dim = 28*28                # used by readers to treat input data as a vector
num_output_classes = 10

#we need a real progress printer
progprint = C.logging.progress_print.ProgressPrinter(freq=0,first=2,log_to_file='mnistG.log')
progprint.log('Initializng the Model ' + repr(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))


# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):
    
    ctf = C.io.CTFDeserializer(path, C.io.StreamDefs(
          labels=C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
          features=C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)))
                          
    return C.io.MinibatchSource(ctf,
        randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

data_found=False # A flag to indicate if train/test data found in local cache
for data_dir in [os.path.join("C:\local\cntk-2-0\cntk\Scripts\CNTK-Samples-2-0", "Examples", "Image", "DataSets", "MNIST"),
                 os.path.join("data", "MNIST")]:
    
    train_file=os.path.join(data_dir, "Train-28x28_cntk_text.txt")
    test_file=os.path.join(data_dir, "Test-28x28_cntk_text.txt")
    
    if os.path.isfile(train_file) and os.path.isfile(test_file):
        data_found=True
        break
        
if not data_found:
    raise ValueError("Please generate the data by completing CNTK 103 Part A")
    
print("Data directory is {0}".format(data_dir))

#prep our input/output containers/shapes
x = C.input_variable(input_dim_model)
y = C.input_variable(num_output_classes)

def create_criterion_function(model, labels):
    loss = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error(model, labels)
    return loss, errs # (model, labels) -> (loss, error metric)

# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=5):
    if len(a) < w:
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))
        
    return mb, training_loss, eval_error


def train_test(train_reader, test_reader, model_func, num_sweeps_to_train_with=10):  #sweeps was 10
    
    # Instantiate the model function; x is the input (feature) variable 
    # We will scale the input image pixels within 0-1 range by dividing all input value by 255.
    model = model_func(x/255)
    
    # Instantiate the loss and error function
    loss, label_error = create_criterion_function(model, y)
    
    # Instantiate the trainer object to drive the model training
    learning_rate = 0.2
    lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
    learner = C.sgd(z.parameters, lr_schedule)
    trainer = C.Trainer(z, (loss, label_error), [learner],[progprint]) #bind up trainer, learner,model,loss,printer
    
    
    # Initialize the parameters for the trainer
    minibatch_size = 64           #was 64
    num_samples_per_sweep = 60000 #was 60000 - use 6000 for demos
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    
    # Map the data streams to the input and labels.
    input_map={
        y  : train_reader.streams.labels,
        x  : train_reader.streams.features
    } 
    
    # Uncomment below for more detailed logging
    training_progress_output_freq = 500
     
    # Start a timer
    start = time.time()

    for i in range(0, int(num_minibatches_to_train)):
        # Read a mini batch from the training data file
        data=train_reader.next_minibatch(minibatch_size, input_map=input_map) 
        trainer.train_minibatch(data)
        print_training_progress(trainer, i, training_progress_output_freq, verbose=1)
    # Print training time
    print("Training took {:.1f} sec".format(time.time() - start))
    progprint.log('Training done - logging model prior to testing')
    
    cntkfuncs = C.logging.graph.depth_first_search(trainer.model,lambda j: (True),depth=-1)
    for afunc in cntkfuncs:
        progprint.log("Name: "+ afunc.name + " Val: " + repr(afunc))    # + "Type: "+ str(type(afunc)))
    # Test the model
    test_input_map = {
        y  : test_reader.streams.labels,
        x  : test_reader.streams.features
    }

    # Test data for trained model
    test_minibatch_size = 512 #was 512
    num_samples = 10000 #was 10000 - make it 1000 for demos 
    num_minibatches_to_test = num_samples // test_minibatch_size

    test_result = 0.0   

    for i in range(num_minibatches_to_test):
    
        # We are loading test data in batches specified by test_minibatch_size
        # Each data point in the minibatch is a MNIST digit image of 784 dimensions 
        # with one pixel per dimension that we will encode / decode with the 
        # trained model.
        data = test_reader.next_minibatch(test_minibatch_size, input_map=test_input_map)
        eval_error = trainer.test_minibatch(data)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))

    #add do traintest
def do_train_test():
        global z
        z = create_model(x)
        print("the z network model is ",repr(z))
        print("model type is: ",type(z))
        reader_train = create_reader(train_file, True, input_dim, num_output_classes)
        reader_test = create_reader(test_file, False, input_dim, num_output_classes)
        train_test(reader_train, reader_test, z)


# function to build model graph API approach
def create_model(features):
    #the defaults can be overridden at layer level
    with C.layers.default_options(init = C.layers.glorot_uniform(), activation = C.relu):
            h = features
            
            h = C.layers.Convolution2D(filter_shape=(5,5),  #was 5,5
                                       num_filters=8,       #was 8 
                                       strides=(1,1),
                                       #activation=C.sigmoid, #jw testing 8/15 - high error rate 88%
                                       pad=True, name="First_Conv")(h)
            #print("first conv layer: ",h)
            h = C.layers.MaxPooling(filter_shape=(2,2), 
                                    strides=(2,2), name="First_Max")(h)
            #print("max_pool:",repr(h))
            h = C.layers.Convolution2D(filter_shape=(3,3),  #was 5,5
                                       num_filters=16,      #was 16
                                       strides=(1,1),       #was (1,1)
                                       pad=True, name="Second_Conv")(h) #comment out the (h) and run this 
            #print("second conv layer:",repr(h))
            #h = C.layers.Convolution2D(filter_shape=(5,5),
            #                          num_filters=24,
            #                          strides=(1,1),
            #                          pad=True,name="third_conv") (h)
            #print("third conv layer",repr(h))
            #this third layer above was added 8/9 2:25PM for testing only
            h = C.layers.MaxPooling(filter_shape=(3,3), 
                                    strides=(3,3), name="Second_Max") (h)
            r = C.layers.Dropout(0.5,name='DropOutB4Dense96')(h) #tests at 0.88 error
            #add a dense layer for testing - 8/14 - quiz should the droput be defined before or after the dense layer?
            r = C.layers.Dense(96,name='Dense96')(r)    # add a layer of 96 densely connection nodes
            #r = C.layers.layers.Dropout(0.5,name='DropOutAfterDense96')(r)   # error is 1.08...
            r = C.layers.Dense(num_output_classes,activation=None,name='Classifier')(r)
            #r = C.layers.layers.Dropout(0.5,name='DropOutAfterDenseClassifier')(r) #jw 8/14 0.39 avg test error
            #print("\n final network",repr(r)) #add 8/8 1:00PM - shows droput
            cntkfunclist = C.logging.graph.depth_first_search(r,lambda j: (True),depth=-1)
            print("CNTK Objects/Functions Dump")
            for afunc in cntkfunclist:
                print("name\t",afunc.name +" Details:" + repr(afunc)+ ":"+ str(type(afunc)))
            #get the weights
            progprint.log("FirstConv Weight: "+ repr(r.First_Conv.W.value)) #show how to get a layers W values
            #print("start:\t",repr(C.logging.find_by_name(r,'first_conv',depth=-1)))
            #firstconvfunc= C.logging.find_by_name(r,'first_conv',depth=-1)
            #print("first conv outputs:\t",firstconvfunc.outputs)
            #print("first conv block arg map:\t",firstconvfunc.block_arguments_mapping)
            print('---------------------------------------------------------------------')
            #print("next:\t",repr(C.logging.find_by_name(r,'first_max',depth=-1)))
            #print("next:\t",repr(C.logging.find_by_name(r,'second_conv',depth=-1)))
            #print("next:\t",repr(C.logging.find_by_name(r,'second_max',depth=-1)))
            #print("next:\t",repr(C.logging.find_by_name(r,'96',depth=-1)))
            #print('----------------------------------------------------------------------')
            
            print("final model:\n ",repr(r))            
            #r=C.debugging.debug_model(r) - not good wo terminal
            return r
        
do_train_test()


