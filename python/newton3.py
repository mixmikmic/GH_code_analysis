from numpy import array,zeros,exp
from numpy.linalg import norm,solve

def f(x):
    y = zeros(3)
    y[0] = x[0]*x[1] - x[2]**2 - 1.0
    y[1] = x[0]*x[1]*x[2] - x[0]**2 + x[1]**2 - 2.0
    y[2] = exp(x[0]) - exp(x[1]) + x[2] - 3.0
    return y

def df(x):
    y = array([[x[1],               x[0],               -2.0*x[2]],                [x[1]*x[2]-2.0*x[0], x[0]*x[2]+2.0*x[1], x[0]*x[1]],                [exp(x[0]),          -exp(x[1]),        1.0]])
    return y

def newton(fun,dfun,x0,M=100,eps=1.0e-14,debug=False):
    x = x0
    for i in range(M):
        g = fun(x)
        J = dfun(x)
        h = solve(J,-g)
        x = x + h
        if debug:
            print i,x,norm(g)
        if norm(h) < eps * norm(x):
            return x

x0 = array([1.0,1.0,1.0])
x = newton(f,df,x0,debug=True)

