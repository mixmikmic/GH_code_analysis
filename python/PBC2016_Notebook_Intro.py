# for both IPython shell and Jupyter notebook

get_ipython().show_usage()
#import os
#os?
#os??

# for both IPython shell and Jupyter notebook

get_ipython().magic('quickref')

# suppress output with a semicolon
2+10;

_ + 10

Out
#Out[4]
#_4

print ('last output:' + _)
print ('next one   :' + __)
print ('and next   :' + ___)

In[2]
_i
_ii

print ('last input:' + _i)
print ('next one  :' + _ii)
print ('and next  :' + _iii)

get_ipython().magic('history')

# Linux/Mac
get_ipython().system('ls')

# Windows
get_ipython().system('dir')

# magic functions
get_ipython().magic('magic')

get_ipython().run_cell_magic('file', 'test.txt', 'Here is my test file and contents.\n\nIt can contain anything.')

with open('test.txt') as f:
    print f.read()

