# Import Node and Function module
from nipype import Node, Function

# Create a small example function
def add_two(x_input):
    return x_input + 2

# Create Node
addtwo = Node(Function(input_names=["x_input"],
                       output_names=["val_output"],
                       function=add_two),
              name='add_node')

addtwo.inputs.x_input =4
addtwo.run()
addtwo.result.outputs

from nipype import Node, Function

# Create the Function object
def get_random_array(array_shape):

    # Import random function
    from numpy.random import random
   
    return random(array_shape)

# Create Function Node that executes get_random_array
rndArray = Node(Function(input_names=["array_shape"],
                         output_names=["random_array"],
                         function=get_random_array),
                name='rndArray_node')

# Specify the array_shape of the random array
rndArray.inputs.array_shape = (3, 3)

# Run node
rndArray.run()

# Print output
print(rndArray.result.outputs)

from nipype import Node, Function

# Import random function
from numpy.random import random


# Create the Function object
def get_random_array(array_shape):
  
    return random(array_shape)

# Create Function Node that executes get_random_array
rndArray = Node(Function(input_names=["array_shape"],
                         output_names=["random_array"],
                         function=get_random_array),
                name='rndArray_node')

# Specify the array_shape of the random array
rndArray.inputs.array_shape = (3, 3)

# Run node
rndArray.run()

# Print output
print(rndArray.result.outputs)

