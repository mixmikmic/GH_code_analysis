# Import modules
import traceback
import math
import pandas as pd
import numpy as np
import scipy
import scipy.linalg as LA
import matplotlib.pyplot as plt

def bisect(f,a,b,tol):
    """
    Computes approximate solution of f(x)=0
    
    Args:
        f (function prototype) : function handle 
        a (real number) : left bound of the interval
        b (real number) : right bound of the interval
        tol (real number) : tolerance
        
    Returns:
        Approximate solution x
        
    Raises:
        ValueError:
            - a * b must be smaller than zero
            - a > b will be considered to be wrong
    """
    try:
        if a > b :
            raise ValueError('a must be <= b')
        
        fa = f(a)
        fb = f(b)
        
        if np.sign(fa) * np.sign(fb) >= 0 :
            raise ValueError('It must be verified that f(a) * f(b) < 0')
            
        while (b - a) / 2 > tol :
            # Find the intermediate point  
            c = (a + b) / 2
            fc = f(c)
            if fc == 0 :
                return c
            elif fa * fc < 0 :
                b = c
                fb = fc
            else :
                a = c
                fa = fc
                
        return (a + b) / 2
            
    except ValueError as e :
        print('ValueError Exception : ', e)
        traceback.print_exception()
    

f = lambda x : math.pow(x,3) + math.pow(x,1) - 1
tolerance = 0.0005
xc = bisect(f,0,1,tolerance)
print('%f ~ %f' %(xc - tolerance, xc + tolerance))

f = lambda x : math.cos(x) - x
tolerance = 1e-7
xc = bisect(f,0,1,tolerance)
print(round(xc,6))

fa = lambda x : math.pow(x,5) + x - 1
fa_xc = bisect(fa,0,1,1e-9)
print('(a) x =',round(fa_xc,8))
fb = lambda x : math.sin(x) - 6 * x - 5
fb_xc = bisect(fb,-1,0.5,1e-9)
print('(b) x =',round(fb_xc,8))
fc = lambda x : math.log(x) + math.pow(x,2) - 3
fc_xc = bisect(fc,1,2,1e-9)
print('(c) x =',round(fc_xc,8))

def generate_points(f,xs):
    ys = np.zeros(xs.size)
    for i in range(len(xs)):
        ys[i] = f(xs[i])
    return ys

fa = lambda x : 2 * math.pow(x,3) + 6 * x - 1
xs = np.linspace(0,1,21)
ys = generate_points(fa,xs)
plt.plot(xs,ys)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.grid(True)
plt.show()

fa_xc = bisect(fa,0,1,1e-7)
print('x =',round(fa_xc,6))

fb = lambda x : math.exp(x - 2) + math.pow(x,3) - x
xs = np.linspace(-1.5,1,31)
ys = generate_points(fb,xs)

plt.plot(xs,ys)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.grid(True)
plt.show()

fb_xc = bisect(fb,-1.5,-0.5,1e-7)
print('x =',round(fb_xc,6))
fb_xc = bisect(fb,-0.5,0.5,1e-7)
print('x =',round(fb_xc,6))
fb_xc = bisect(fb,0.5,1.5,1e-7)
print('x =',round(fb_xc,6))

fc = lambda x : 1 + 5 * x - 6 * math.pow(x,3) + math.exp(2 * x) 
xs = np.linspace(-1,0,51)
ys = generate_points(fc,xs)

plt.plot(xs,ys)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.grid(True)
plt.show()

fc_xc = bisect(fc,-0.5,0.5,1e-7)
print('x =',round(fc_xc,6))
fc_xc = bisect(fc,-1.5,-0.5,1e-7)
print('x =',round(fc_xc,6))

def rev_hilbert_matrix_eigvals(upper_left_element):
    hilbert_matrix = LA.hilbert(5)
    hilbert_matrix[0][0] = upper_left_element
    return LA.eigvals(hilbert_matrix)

f = lambda x : np.max(rev_hilbert_matrix_eigvals(x)).real - scipy.pi
xs = np.linspace(2,4,21)
ys = generate_points(f,xs)

plt.plot(xs,ys)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.grid(True)
plt.show()

xc = bisect(f,2.5,3.5,1e-7)
print('Make A[1,1] to be',round(xc,6),'to make the largest eigenvalue equal to pi')

def fpi(g, x0, k, tol = 0):
    """
    Computes approximate solution of g(x)=x by Fixed-Point Iteration
    
    Args:
        g (function prototype) : function handle
        x0 (real number) : starting guess 
        k (integer) : number of iteration steps
        tol (real number) : tolerance (default : 0)
        
    Returns:
        Approximate solution xc
    
    Raises:
        ValueError :
            - g is None
            - k is not the type of positive integer 
            - tol is negative
    """
    try:
        if g is None :
            raise ValueError('f is null')
        if k != int(k) :
            raise ValueError('k must be type of integer')
        if k < 0 :
            raise ValueError('k must be positive integer')
        if tol < 0 :
            raise ValueError('tol must be positive')
            
        x = x0
        for _ in range(k) :
            tx = g(x)
            if tol > 0 and abs(tx - x) / abs(tx) < tol :
                return tx
            else :
                x = tx
        
        return x
    except ValueError as e:
        print('ValueError Exception : ', e)
        traceback.print_exception()

"""
cosx = sinx
=> x + cosx - sinx = x
=> g(x) = x + cosx - sinx
"""
g = lambda x : x + math.cos(x) - math.sin(x)
x = np.arange(21)
df = pd.DataFrame()

df['g(x)']= [fpi(g, 0, x[i]) for i in range(21)]

print(df)

g = lambda x : 2.8 * x - math.pow(x,2)
print(fpi(g, 0.1, 300))

"""
(a) x^3 = 2x + 2
==> x = (2x + 2)^(1/3)
==> g(x) = (2x + 2)^(1/3)
"""
a = lambda x : math.pow(2 * x + 2, 1 / 3)
print(fpi(a,0,100,1e-8))

"""
(b) e^x + x = 7
==> e^x = 7 - x
==> ln(7 - x) = x
==> g(x) = ln(7 - x)
"""
b = lambda x : math.log(7 - x)
print(fpi(b,0,200,1e-8))

"""
(c) e^x + sinx = 4
==> e^x = 4 - sinx
==> x = ln(4 - sinx)
"""
c = lambda x : math.log(4 - math.sin(x))
print(fpi(c,0,100,1e-8))

from scipy.optimize import fsolve
f = lambda x : math.sin(x) - x
r = fsolve(f,0.1)[0]
print('forward error = %.16f' %(abs(r)))
print('backward error = %.16f' %(abs(f(r))))

from scipy.optimize import fsolve
f = lambda x : 2 * x * math.cos(x) - 2 * x + math.sin(math.pow(x, 3))
r = fsolve(f,0.04)[0]
print('x = %.16f' % r)
print('forward  error : %e' % abs(r) )
print('backward error : %e' % abs(f(r)) )

"""
f(x) 
=> 2x * (cos(x) - 1) + sin(x^3)
=> (2x * (cos^2(x) - 1) + (cos(x) + 1) * sin(x^3)) / (cos(x) + 1)
=> (2x * -sin^2(x)  + (cos(x) + 1) * sin(x^3)) / (cos(x) + 1)
"""
f = lambda x : (2 * x * -math.pow(math.sin(x), 2) + (math.cos(x) + 1) * math.sin(math.pow(x, 3))) / (math.cos(x) + 1)
r = bisect(f,-0.1,0.2,1e-7)
print('x = %.16f' % r)
print('forward  error : %e' % abs(r) )
print('backward error : %e' % abs(f(r)) )

def newton_method( f, df, x0, k = 500, tol = 1e-6) :
    """
    Use Newton's method to find the root of the function
    
    Args:
        f (function prototype) : function handle
        df (function prototype) : derivative function handle
        x0 (real number) : starting guess 
        k (integer) : number of iteration steps (default : 500)
        tol (real number) : tolerance (default : 1e-6)
        
    Return:
        Approximate solution xc
        
    Raises:
        ValueError :
            - f or df is None
            - k is smaller than 0
    """
    try:
        if f is None or df is None :
            raise ValueError('Function handle f or df is Null')
        
        if k <= 0 :
            raise ValueError('Iteration k must be larger than 0')
        
        xc = x0
        for _ in range(k) :
            xt = xc - f(xc) / df(xc)
            if tol > 0 and abs(xt - xc) / abs(xc) < tol :
                return xt
            else :
                xc = xt
        return xc
    except ValueError as e :
        print('ValueError Exception : ', e)
        traceback.print_exception()   

f = lambda x : math.pow(x,3) + x - 1
df = lambda x : 3 * math.pow(x,2) + 1
xc = newton_method(f, df, -0.7, 25, 1e-8)
print ('x = %.8f' % xc)

def secant_method(f, x0, x1, k=500) :
    """
    Use Secant's method to find the root of the formula
    
    Args:
        f (function prototype) : function handle
        x0 (real number) : initial guess
        x1 (real number) : initial guess 
        k (integer) : number of iteration steps (default : 500)
        
    Return:
        Approximate solution xc
        
    Raises:
    
    """
    x = np.zeros(k)
    x[0] = x0
    x[1] = x1
    for i in range(1,k - 1):
        if f(x[i]) - f(x[i - 1]) == 0 :
            return x[i]
        x[i + 1] = x[i] - f(x[i]) * (x[i] - x[i - 1])  / (f(x[i]) - f(x[i - 1]))
        
    return x[k - 1]

f = lambda x : math.pow(x, 3) + x - 1
secant_method(f, 0, 1, 100)

def false_position_method(f, a, b, k) :
    """
    Use false position method to find the root of the formula f
    
    Args:
        f (function prototype) : function handle
        a (real number) : the lowerbound of the bracket of initial guess
        b (real number) : the upperbound of the bracket of initial guess
        k (integer) : number of iteration steps (default : 500)
        
    Return:
        Approximate solution xc
        
    Raises:
        ValueError:
            - f(a) * f(b) must be smaller than 0 (exclude zero)
        
    """
    if f(a) * f(b) >= 0 :
        raise ValueError('f(a) * f(b) must be < 0')
    for _ in range(k) :
        c = (b * f(a) - a * f(b)) / (f(a) - f(b))
        if f(c) == 0 :
            return c
        if f(a) * f(c) < 0 :
            b = c
        else :
            a = c
    return c

f = lambda x : pow(x, 3) - 2 * pow(x, 2) + 1.5 * x
false_position_method(f, -1, 1, 100)

def inverse_quadratic_interpolation(f, x0, x1, x2, k):
    a = x0
    b = x1
    c = x2
    for _ in range(k) :
        q = f(a) / f(b)
        r = f(c) / f(b)
        s = f(c) / f(a)
        
        denominator = (q - 1) * (r - 1) * (s - 1)
        if denominator == 0 :
            break
        
        tmp = c - (r * (r - q) * (c - b) + (1 - r) * s * (c - a)) / denominator
        a = b
        b = c
        c = tmp
        
        
    return c

f = lambda x : pow(x, 3) + x - 1
inverse_quadratic_interpolation(f, 0, 0.5, 1, 20)

