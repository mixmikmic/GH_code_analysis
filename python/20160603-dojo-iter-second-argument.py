from functools import partial

for foo in iter(partial(input), 'quit'):
    print('in loop:', foo)
    
print('after loop:', foo)

