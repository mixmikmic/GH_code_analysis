get_ipython().magic('load_ext line_profiler')

def example_function(myRange):
    # directly convert range to string list
    str_list = []
    for i in myRange:
        str_list.append(str(i))
        
def example_function2(myRange):
    # use list comprehension to convert range to string list
    str_list = [str(i) for i in myRange] 
        

get_ipython().magic('lprun -f example_function example_function(range(1000000))')

get_ipython().magic('lprun -f example_function2 example_function2(range(1000000))')

import numpy as np

def fib(A):
    '''
    CALCULATE FIRST N FIBONACCI NUMBERS
    '''
    n = len(A)
    
    for i in range(n):
        if i == 0:
            A[i] = 0.
        elif i == 1:
            A[i] = 1.
        else:
            A[i] = A[i-1] + A[i-2]
            
    return A

dat_in = np.zeros(10) 
dat_out = fib(dat_in)
dat_out

dat_in = np.zeros(1000) 

get_ipython().magic('lprun -f fib fib(dat_in)')

get_ipython().system('ls')

get_ipython().run_cell_magic('writefile', 'fib1.f', 'C FILE: FIB1.F\n      SUBROUTINE FIB(A,N)\nC\nC     CALCULATE FIRST N FIBONACCI NUMBERS\nC\n      INTEGER N\n      REAL*8 A(N)\n      DO I=1,N\n         IF (I.EQ.1) THEN\n            A(I) = 0.0D0\n         ELSEIF (I.EQ.2) THEN\n            A(I) = 1.0D0\n         ELSE \n            A(I) = A(I-1) + A(I-2)\n         ENDIF\n      ENDDO\n      END\nC END FILE FIB1.F')

get_ipython().system('f2py -c fib1.f -m fib1')

get_ipython().system('ls')

import fib1
import numpy as np

print fib1.fib.__doc__

a = np.zeros(9)

fib1.fib(a)

a

a = np.zeros(1000)

get_ipython().magic('timeit fib1.fib(a)')

get_ipython().magic('timeit fib(a)')

