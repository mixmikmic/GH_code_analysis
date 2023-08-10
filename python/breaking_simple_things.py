from battle_tested import fuzz, battle_tested

def add_to_hello(a):
    return 'hello '+a
add_to_hello('world')

fuzz(add_to_hello, seconds=1, keep_testing=True)

def add_to_hello_with_format(a):
    return 'hello {}'.format(a)
add_to_hello_with_format('world')

fuzz(add_to_hello_with_format, seconds=1, keep_testing=True)

def count(a):
    return len(a)
count('hello')

fuzz(count, seconds=1, keep_testing=True)



