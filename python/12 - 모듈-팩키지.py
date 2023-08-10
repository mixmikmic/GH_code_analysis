import mymodule

mymodule.mysum(1, 2)

mymodule.mymultiply(2, 3)

from mymodule import mysum, mymultiply

mysum(1, 2)

mymultiply(2, 3)

import time
print ('hello')
time.sleep(3)
print ('world')

from mylib.math import mysum4

mysum4(1, 2, 3, 4)

from mylib import mysum4 # mylib/__init__.py 내 mysum4

mysum4(1, 2, 3, 4)

import sys

sys.path

import math

math.__file__

import askdjango # error

from mylib.math import get_file

get_file()

from os.path import dirname

dirname(get_file())

dirname(dirname(get_file()))





