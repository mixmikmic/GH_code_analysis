# simple function to add two numbers
def sum_two_numbers(a, b):
    return a + b

# after this line x will hold the value 3!
x = sum_two_numbers(1,2)
print x

def sum_two_numbers(a, b = 10):
    return a + b

print sum_two_numbers(10)
print sum_two_numbers(10, 5)

# Global variable
a = 10 

# Simple function to add two numbers
def sum_two_numbers(b):
    return a + b

# Call the function and print result
print sum_two_numbers(10)

# Simple function to add two number with b having default value of 10
def sum_two_numbers(a, b = 10):
    return a + b

# Call the function and print result
print sum_two_numbers(10)

print sum_two_numbers(10, 5)

# Simple function to loop through arguments and print them
def foo(*args):
    for a in args:
        print a

# Call the function
foo(1,2,3) 

# Simple function to loop through arguments and print them
def foo(**kwargs):
    for a in kwargs:
        print a, kwargs[a]
        

# Call the function
foo(name='John', age=27)

