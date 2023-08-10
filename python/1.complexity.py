# Global variables

# The number of watts used by the computer we're running experiments on.  You can use the default value below,
# or look up the correct value for the computer you're using and replace the value below with that.
COMPUTER_ENERGY_WATTS = 200.0

def execute_another_function(function, function_input):
    """
    A function that takes another function as input, along with the single parameter to that function, and
    returns the result of running the given function on the given input.
    """
    return function(function_input)

def function_to_send(x):
    return x + 2

print execute_another_function(function_to_send, 5)

