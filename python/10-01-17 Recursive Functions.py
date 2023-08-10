print(5*4*3*2*1)

# Create a function inputting n, that, 
def factorial(n):
    # if n is less than or equal to 1, 
    if n <= 1:
        # return n, 
        return n
    
    # if not, return n multiplied by the output
    # of the factorial function of one less than n
    return n*factorial(n-1)

# run the function
factorial(5)

