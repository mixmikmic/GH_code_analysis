def function_of_no_return(a):
    print ("Parameter passed to the function is {0}".format(a))
    
x = function_of_no_return(42)
print ('Function returns', x)

def function_of_print_return(a):
    return print ("Parameter passed to the function is {0}".format(a))
    
x = function_of_print_return(42)
print ('Function returns', x)

def proper_function(a):
    print ("Parameter passed to the function is {0}".format(a))
    return a
    
x = proper_function(42)
print ('Function returns', x)

def half(num):
    h = num/2
    return h

h_true = 2

h = half(4)
print("half(4) -> {} [True: {}]".format(h, h_true))
assert h == h_true
assert type(h) == int

h = half(5)
print("half(5) -> {} [True: {}]".format(h, h_true))
assert h == h_true
assert type(h) == int

print("\n(Passed!)")

