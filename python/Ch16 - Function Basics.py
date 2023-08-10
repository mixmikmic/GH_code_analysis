# Assign a function to another name
def func():
    print("This is a function")
    
another_name = func
another_name()

# Passed by object reference

nums = [1, 2, 3]

def append_four(numbers):
    numbers.append(4)
    
append_four(nums)
print(nums)

