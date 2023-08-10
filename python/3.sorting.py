def default_params(x = 3, y = None):
    """
    An example function with default parameters.  Try calling it to see what happens!
    """
    if y == None:
        return x * 2
    else:
        return x * y

list = [3, 5, 2, 8, 1, 1, 2]
print sorted(list)
print list

tuple = ("b", "a", "c", "f", "A")
print sorted(tuple)
print tuple

dict = {1:"r", 2:"s", 5:"e", 3:"s"}
print sorted(dict)
print dict

dict1 = {"special key":1, "a":3, "b":5}
dict2 = {"special key":3, "a":3, "c":7}
dict3 = {"special key":2, "a":2, "b":3}
list_of_dicts = [dict1, dict2, dict3]

def get_key_to_compare(dict):
    return dict["special key"]

print sorted(list_of_dicts, key = get_key_to_compare)

