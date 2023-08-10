import json 
def print_dict(dd): 
    print json.dumps(dd, indent=2)

d1 = dict() 
d2 = {}

print_dict(d1)
print_dict(d2)

d3 = {
    'one': 1, 
    'two': 2
}

print_dict(d3)

d4 = dict(one=1, two=2)

print_dict(d4)

keys = ['one', 'two', 'three']
values = [1, 2, 3]

d5 = {key: value for key, value in zip(keys, values)}
print_dict(d5)

d1['key_1'] = 1
d1['key_2'] = False

print_dict(d1)

d1['list_key'] = [1, 2, 3]
print_dict(d1)

d1['dict_key'] = {'one': 1, 'two': 2}
print_dict(d1)

del d1['key_1']
print_dict(d1)

print d1.keys() 

for item in d1:
    print item

d1['dict_key']['one']

for key, value in d1.items(): 
    print key, value

for key, value in d1.iteritems(): 
    print key, value 

print d1.keys() 
print d1.values() 

def dict_only((key, value)): 
    return type(value) is dict

print 'All dictionary elements:'
print filter(dict_only, d1.items())

print 'Same as above, but with inline function (lambda):'
print filter(lambda (key, value): type(value) is dict, d1.items())

























