from numpy import abs

def f(x):
    return x**5 + 1.0

def df(x):
    return 5.0*x**4

def newton(f,df,x0,M=100,eps=1.0e-15):
    x = x0
    for i in range(M):
        dx = - f(x)/df(x)
        x  = x + dx
        print i,x,abs(f(x))
        if abs(dx) < eps * abs(x):
            return x

x0 = 1.0 + 1.0j
x = newton(f,df,x0)

