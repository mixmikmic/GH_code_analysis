help(sum)

L = [2,3,4]
sum(L)

get_ipython().magic('pinfo sum')

get_ipython().magic('pinfo L')

get_ipython().magic('pinfo L.reverse')

def square(a):
    """Return the square of a."""
    return a ** 2

get_ipython().magic('pinfo2 square')

5+5

6+6

7+7

print(_)
print(__)
print(___)

In

In[1]

Out

type(Out)

Out[2]

_2 # Shortcut for Out[2]

exec(In[6]) # This executes a particular string

get_ipython().magic('lsmagic # Lists available magics')

get_ipython().magic('pinfo %whos')

get_ipython().run_cell_magic('latex', '', "\\[\\mathcal{L}=E_s\\sum_{t=0}^\\infty \\beta^t\n\\left\\{ F(x_{s+t},u_{s+t}) + \\lambda'_{s+t}\n(f_{s+t}(x_{s+t},u_{s+t},\\epsilon_{s+t})-x_{s+t+1}) \\right\\}\\]")

get_ipython().run_cell_magic('writefile', 'blabla.txt', 'This is some garbage text.\nThat will be saved to a file.')

get_ipython().system('dir # list the contents of the current directory')

get_ipython().system('ver # print the OS version')

var = get_ipython().getoutput('ver')
var

f1 = lambda a, b: a / b

def f2(x):
    a = x
    b = x - 5
    return f1(a, b)

get_ipython().magic('xmode Context')
f2(5)

get_ipython().magic('xmode Plain')
f2(5)

get_ipython().magic('xmode Verbose')
f2(5)

get_ipython().magic('xmode Context')

f2(5)

get_ipython().magic('debug')

get_ipython().run_cell_magic('writefile', 'cw.py', 'x = [1,2,3,2,5]\ndef ComputeWeights(x):\n    S = sum(x)\n    Wt = [i//S for i in x] # note the mistake in using // instead of /\n    return Wt\n\nprint(ComputeWeights(x))')

# Run this several times and input 5, -5 and aaa
from math import sqrt
inp = input("Input number: ")
try:
    result = sqrt(float(inp))
    print("The square root of %s is %f"%(inp,result))
except:
    print("There was a problem computing the square root of %s"%(inp))

from math import sqrt
import sys
try:
    del result # Remove residual results if they exist
except NameError:
    pass # do nothing
inp = input("Input number: ")
try:
    try:
        try:
            num = float(inp)
        except ValueError:
            print("Are you sure you inputted a number?")
            sys.exit() # We need this to avoid falling through
                       # to the next exception. Comment it out 
                       # to see what happens when you input 'aaa'
        result = sqrt(num)
        print("The square root of %f is %f"%(num,result))
    except ValueError:
        print("Your number is negative!")
except SystemExit:
    pass # This serves to ensure a silent System exit

try:
    print("try something here")
except:
    print("this happens only if it fails")
else:
    print("this happens only if it succeeds")
finally:
    print("this happens no matter what (usually used to clean up)")

def divXY(x,y):
    if y == 0:
        raise ValueError("Are you kidding? You can't divide by zero!")
    else: 
        return x/y

divXY(2,0)

# This times a one-line statement, executing it once
get_ipython().magic('time L = [i for i in range(1000000)]')

get_ipython().run_cell_magic('time', '', '# This times an entire cell, executing it once\n\nS = [str(i) for i in range(1000000)]\nL = [i for i in range(1000000)]\ndel S,L')

# This times a one-line statement, executing it multiple times
get_ipython().magic('timeit L = [i for i in range(1000000)]')

get_ipython().run_cell_magic('timeit', '', '# This times an entire cell, executing it multiple times\n\nS = [str(i) for i in range(1000000)]\nL = [i for i in range(1000000)]\ndel S,L')

get_ipython().magic('prun S = [str(i) for i in range(1000000)]')

get_ipython().run_cell_magic('prun', '', 'def sum_of_lists(N):\n    total = 0\n    for i in range(5):\n        L = [j ^ (j >> i) for j in range(N)]\n        total += sum(L)\n    return total\n\nsum_of_lists(1000000)')

