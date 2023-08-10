import IPython
import numpy as np
from sys import version 
print ' Reproducibility conditions for this notebook '.center(85,'-')
print ('Python version:     ' + version).center(85)
print ('Numpy version:      ' + np.__version__)
print 'IPython version:    ' + IPython.__version__
print '-'*85

import datetime
from my_package import suppress_errors

@suppress_errors
def log_error(message, log_file='errors.log'):
    """Log an error message to a file."""
    log = open(log_file,'w')
    log.write('%s\t%s\n' % (datetime.datetime.now(), message))

import datetime
from my_package import suppress_errors

def log_error(message, log_file='errors.log'):
    """Log an error message to a file."""
    log = open(log_file, 'w')
    log.write('%s\t%s\n' % (datetime.datetime.now(), message))
    
log_error = suppress_errors(log_error)

def multiply_by(factor):
    '''Return a function that multiplies values by the given factor'''
    def multiply(value):
        '''Multiply the given value by the factor already provided'''
        return value * factor
    return multiply

times2 = multiply_by(2)
print(times2(2))

times2 = multiply_by(2)
print times2(5)
print times2(10)
print times2(100)

times10 = multiply_by(10)
print times10(5)
print times10(10)
print times10(100)

def suppress_errors(func):
    """Automatically silence any errors that occur within a function"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
        
    return wrapper

import functools
def suppress_errors(func):
    """Automatically silence any errors that occur within a function"""
    @functools.wrapper(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            pass

    return wrapper

import datetime
import os
import time
from my_package import suppress_errors

@suppress_errors()
def process_updated_files(directory, process, since=None):
    """
    Processes any new files in a `directory` using the `process` function.
    If provided, `since` is a date after which files are considered updated.

    The process function passed in must accept a single argument: the absolute
    path to the file that needs to be processed.
    """
    
    if since is not None:
        # Get a threshold that we can compare to the modification time later
        threshold = time.mktime(since.timetuple()) + since.microsecond / 1000000
    else:
        threshold = 0
        
    for filename in os.listdir(directory):
        path = os.path.abspath(os.path.join(directory, filename))
        if os.stat(path).st_mtime > threshold:
            process(path)

def memoize(func):
    """
    Cache the results of the function so it doesn't need to be called
    again, if the same arguments are provided a second time.
    """
    cache ={}
    @functools.wraps(func)
    def wrapper(*args):
        if args in cache:
            return cache[args]
        
        print('First time calling %s()' % func.__name__)
        
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

@memoize
def multiply(x, y):
    return x * y

multiply(6, 7)

multiply(6, 7)

@memoize
def factorial(x):
    result = 1
    for i in range(x):
        result *= i+1
    return result

factorial(10)

factorial(10)

import collections
import functools

class memoized(object):
    def __init__(self, func):
        self.func = func
        self.cache = {}    
    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable: a dict, or a list, for instance.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            print('First time calling %s()' % self.func.__name__)
            self.cache[args] = value
            return value
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)

@memoized
def factorial(x):
    result = 1
    for i in range(x):
        result *= i+1
    return result

factorial(10)

factorial(10)



