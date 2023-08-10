import random

def get_uniform(low, high):
    num = random.uniform(low, high)
    return num
    
def num_to_str(num):
    print type(num)
    str_num = "number:" + str(num) + "\n"
    print type(str_num)
    return str_num
    

