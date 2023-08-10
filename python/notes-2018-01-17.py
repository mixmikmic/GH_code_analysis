def box_surface(w,d,h):
    "Compute surface area of rectangular box with dimensions w, d, and h."
    # Opposite sides have equal area
    area = 2*w*d + 2*w*h + 2*d*h
    return area

# Outside the function definition
# Test our function for values where we know the result:
# box_surface(1,1,1) returns 6
box_surface(1,1,1)

whos

get_ipython().magic('pinfo box_surface')

def mn_integral(m,n,a,b,N):
    '''Compute the (right) Riemann for f(x) = (x^m + 1)/(x^n + 1) on interval [a,b].
    
    Input:
        m, n: numbers
        a,b: numbers, limits of integration
        N: size of partition of interval [a,b]
    Output:
        Compute the (right) Riemann sum of f(x) from a to b using a partition of size N.
    Example:
        >>> mn_integral(0,1,0,1,2)
        1.1666666666666665
    '''
    # Compute the width of each subinterval
    delta_x = (b - a)/N
    
    # Create N+1 evenly space x values from a to b
    xs = [a + k*delta_x for k in range(0,N+1)]
    
    # Compute terms of the sum
    terms = [(xs[k]**m + 1)/(xs[k]**n + 1)*delta_x for k in range(1,N+1)]
    
    # Compute the sum
    riemann_sum = sum(terms)
    
    return riemann_sum

mn_integral(0,1,0,1,2)

7/6

mn_integral(1,2,0,1,100000)

0.5*0.69314718 + 3.14159265/4

