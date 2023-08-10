''' 
Create a function. 

The argument 'text' is the string to print with the default value 'default string'
and the argument 

The argument 'n' is an integer of times to print with the default value of 1. 

The function should return a string.
'''
def print_text(text:'string to print'='default string', n:'integer, times to print'=1) -> str:
    return text * n

# Run the function with arguments
print_text('string',4)

# Run the function with default arguments
print_text()

