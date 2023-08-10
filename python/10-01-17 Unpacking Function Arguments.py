# Create a dictionary of arguments
argument_dict = {'a':'Alpha', 'b':'Bravo'}

# Create a list of argument
argument_list = ['Alpha','Bravo']

# Create a function that take two inputs
def simple_function(a, b):
    # and prints them combined
    return a + b

# Run the fucntion with the unpacked argument dictionary
simple_function(**argument_dict)

# Run the function with the unpacked argument list
simple_function(*argument_list)

