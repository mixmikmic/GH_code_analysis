# Python 3 compat
import sys
if sys.version_info[0] >= 3:
    raw_input = input

name = raw_input("What is your name? ")
name

def div(x, y):
    return x/y

div(1,0)

get_ipython().magic('debug')

