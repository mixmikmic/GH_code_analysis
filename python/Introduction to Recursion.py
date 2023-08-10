def fact(n):
    '''
    Returns factorial of n (n!).
    Note use of recursion
    '''
    # BASE CASE!
    if n == 0:
        print 'reached base case'
        return 1
    
    # Recursion!
    else:
        print n
        return n * fact(n-1)

fact(5)

from IPython.display import Image
from IPython.core.display import HTML 
Image(url= 'http://faculty.cs.niu.edu/~freedman/241/241notes/recur.gif')

