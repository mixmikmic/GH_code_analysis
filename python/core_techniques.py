import collections
import functools
import more_itertools
import json

# start with a function that produces a list of squared numbers
def squares_as_list(max_n):
    accum = []
    x = 1
    while x <= max_n:
        accum.append(x * x)
        x = x + 1
    return accum

# output the result
result = squares_as_list(10)
print('Type is: ' + str(type(result)))
for i in result:
    print(i)

# here is a similar function, but implemented as a generator
def squares_as_generator(max_n):
    x = 1
    while x < max_n:
        yield x * x
        x = x + 1


result = squares_as_generator(10)
print('Type is: ' + str(type(result)))

# # loop directly as an iterable
print('All 10 using a loop')
for s in result:
    print(s)
    
# print('Just 5 iterations to demonstrate deferred evaluation...')
another_gen = squares_as_generator(10)
print(next(another_gen))
print(next(another_gen))
print(next(another_gen))
print(next(another_gen))
print(next(another_gen))

#
# Generator Chaining example
#

def f_A(n):
    x = 1
    while x < n:
        yield x * x
        x = x + 1
        
def f_B(iter_a):
    for y in iter_a:
        yield y + 10000
        
def f_C(iter_b):
    for z in iter_b:
        yield "'myprefix " + str(z) + "'"
        
# chain the first two
gen_a = f_A(10)
gen_b = f_B(gen_a)
print('First two chained')
for r in gen_b:
    print(r)

# print('\nAll 3 chained')
gen_a = f_A(10)
gen_b = f_B(gen_a)
gen_c = f_C(gen_b)
for r in gen_c:
    print(r)


# source: assume this list are the database rows
SOURCE_DATA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]

DESTINATION_DB = collections.OrderedDict()

def extractor(source_data):
    for item in source_data:
        yield item
        
def transformer(iter_extractor):
    for item in iter_extractor:
        # transform it into a tuple of (n, n^2)
        transformed_item = (item, item * item)
        yield transformed_item
        
def loader(iter_transformer, db):
    for item in iter_transformer:
        # insert each tuple as an item into the storage dictionary
        k = str(item[0])
        v = item[1]
        db[k] = v
        

# here is a simple example of chaining generators
extracted_gen = extractor(SOURCE_DATA)

transformed_gen = transformer(extracted_gen)

loader(transformed_gen, DESTINATION_DB)

# output the loaded results
print(json.dumps(DESTINATION_DB, indent=2))

def add(x, y):
    return x + y

# print('Simple addition')
# print('1 + 2 = %d' % add(1, 2))
# print('2 + 3 = %d' % add(2, 3))

print('partial add_1 function')
# NOTE: order of args matters!
add_1 = functools.partial(add, 1)

print('add_1(1) = %d' % add_1(1))
print('add_1(2) = %d' % add_1(2))

print('partial add_2 function')
add_2 = functools.partial(add, 2)

print('add_2(1) = %d' % add_2(1))
print('add_2(2) = %d' % add_2(2))

import functools

# similarly, you can freeze kwargs to avoid ordering constraints
def pow(x, n=1):
    return x ** n
    
print('regular')
print( pow(2, n=3) )

print('partial with n=2')
pow_2 = functools.partial(pow, n=2)
print(type(pow_2))
print( pow_2(2) )

print('partial with n=3')
pow_3 = functools.partial(pow, n=3)
print( pow_3(2) )

pow_easy = functools.partial(pow, 5, n=2)
print( pow_easy() )

# example: this tranformer generator has multiple kwargs which serve
# parameters indicating its behavior
def tranform_func_with_config(
    iter_extractor, 
    translate=0,
    scale=1, 
    cast_func=int
):
    for x in iter_extractor:
        t = x + translate
        t = scale * t
        t = cast_func(t)

        yield (x, t)
        
# now we can create multiple transformer configurations via partial functions
# these configurations can be read from a JSON file
config_1 = {'translate': 1, 'scale': 2}
config_2 = {'scale': -1, 'cast_func': str}

# create partial functions quickly by unpacking the configuration to freeze the kwargs
transform_1 = functools.partial(tranform_func_with_config, **config_1)
transform_2 = functools.partial(tranform_func_with_config, **config_2)

# let's output one of them
extracted_gen = extractor(SOURCE_DATA)
tranform_1_gen = transform_1(extracted_gen)

for t in tranform_1_gen:
    print(t)
    
# any questions?

# the real power is that the partial function _encapsulates_ the confirmation so that 
# other functions (like this simple process method) need not be concerned with it
def process(f_extractor, f_transformer, f_loader):
    
    # run the process
    extractor_gen = f_extractor(SOURCE_DATA)
    
    transformer_gen = f_transformer(extractor_gen)
    
    f_loader(transformer_gen, DESTINATION_DB)


DESTINATION_DB.clear()
print('configuration 1')
process(extractor, transform_1, loader)
print(json.dumps(DESTINATION_DB, indent=2))


DESTINATION_DB.clear()
print('\nconfiguration 2')
process(extractor, transform_2, loader)
print(json.dumps(DESTINATION_DB, indent=2))

# range() is a python built-in.  since python 3, it is a generator!
source_gen = range(20)

print('normal consumption')
for item in source_gen:
    print(item)
    
print('\nbatched consumption')
source_gen = range(20)
chunk_size = 3
batched_gen = more_itertools.chunked(source_gen, chunk_size)
for item in batched_gen:
    print('{} of size {}: {}'.format(type(item), len(item), item))

# node - current node in the tree
# path - list of strings representing 'path components' down the JSON tree

# f_gen_items - produces 'transformed' items for a node
# f_gen_children - produces child nodes to search

def _recursive_map_nested(node:dict, path: list, f_gen_items, f_gen_children):
    if not node:  # empty node
        return


    gen_items = f_gen_items(node, path)
    yield from gen_items

    gen_children = f_gen_children(node, path)
    for child_path, child_node in gen_children:
        yield from _recursive_map_nested(
            child_node,
            child_path,
            gen_items,
            gen_children)

def my_gen_items(node: dict, path: list):
    """ converts scalar dictionary items to response event arguments reflecting answers """
    for k, v in node.items():
        if not isinstance(v, (dict, list,)):
            path_str = '.'.join(path + [k])
            node_info = ...
            
            if node_info:
                str_value= ...
                yield path_str, {
                    'answer_type': node_info.answer_type, 
                    'value': str_value
                }


def my_gen_children():
    """ locates and generates child nodes"""
    node_slug = node.get('slug')
    children = node.get('children')
    if node_slug and children:
        for child in children:
            child_slug = child.get('slug')
            if child_slug:
                yield path + [child_slug], child
    

        
# initial call would be
root = { ... }
transformed_items = _recursive_map_nested(root, [], my_gen_items, my_gen_children)

# pass 'transformed_items' (another generator) to the loader

@functools.lru_cache(maxsize=4)
def cached_pow(x, n):
    print("-- Oh be careful... I'm expensive!")
    return x ** n

# this will run the actual method but cache the results
print('Populate cache with 2 different items')
print( cached_pow(2, 3) )
print( cached_pow(2, 4) )

# this will use cached results (notice the absence of the warning)
print('\nRe-run same requests so that it retrieves from the cache')
print( cached_pow(2, 3) )
print( cached_pow(2, 3) )
print( cached_pow(2, 4) )
print( cached_pow(2, 4) )
print( cached_pow(2, 4) )

# this will force an eviction (2+3 > 4 max items) of the first pow(2,3) result
print('\n3 more different items')
print( cached_pow(2, 5) )
print( cached_pow(2, 6) )
print( cached_pow(2, 7) )

# run the very last one along with (2,3) again to re-evaluate
print('\n(2,3) should have been evicted, will require an evaluation')
print( cached_pow(2, 7) )
print( cached_pow(2, 3) )

print('cache metrics')
cache_info = cached_pow.cache_info()
print(cache_info)

# a contrived session class which uses our contrived database
class CrankySession(object):
    def __init__(self, db):
        self.db = db
    
    def query(self, idx: int):
        print("-- fine fine... I'll check the database")
        return self.db[idx]
    
    def __hash__(self):
        raise RuntimeError("WATCH IT BUDDY! I'm not hashable!")
        
# let's use the lru_cache decorator disregarding the documentation regarding hashable arguments
@functools.lru_cache(maxsize=4)
def broken_session_lookup(session: CrankySession, idx: int):
    return session.query(idx)

# now try running it
session = CrankySession(DESTINATION_DB)
broken_session_lookup(session, "1")

# start with an unwrapped function
def raw_session_lookup(session: CrankySession, idx: int):
    return session.query(idx)

# create a new partial function to "freeze" the session argument
partial_session_lookup = functools.partial(raw_session_lookup, session)

# now you can safely wrap the partial function with the lru_cache method
# NOTE: you need to call the wrapper directly rather than using a decorator syntax
cache_wrapper = functools.lru_cache(maxsize=4)
cached_session_lookup = cache_wrapper(partial_session_lookup)

# now call it to your heart's content
print(cached_session_lookup("1"))
print(cached_session_lookup("2"))
print(cached_session_lookup("2"))
print(cached_session_lookup("1"))
print(cached_session_lookup("1"))
cache_info = cached_session_lookup.cache_info()
print(cache_info)



