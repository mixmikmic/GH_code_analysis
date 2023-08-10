from compecon import MCP, jacobian
# from compecon.tools import jacobian
import numpy as np

def printx(x):
    print(f'\nSolution is [x, y] = [{x[0]:.4f}, {x[1]:.4f}]')
    
x0 = [0.5, 0.5]    

def func(z):
    x, y = z
    return np.array([1 + x*y - 2*x**3 - x, 2*x**2 - y])
                      
F = MCP(func, [0, 0], [1,1])

x = F.zero(x0, transform='minmax', print=True)
printx(x)

# x = F.zero(x0, transform='ssmooth', print=True)
# printx(x)
# FIXME: generates error

def func2(z):
    x, y = z
    f = [1 + x*y - 2*x**3 - x, 2*x**2 - y]
    J = [[y-6*x**2-1, x],[4*x, -1]]
    return np.array(f), np.array(J)                  

F2 = MCP(func2, [0, 0], [1,1])

x = F2.zero(x0, transform='minmax', print=True)
printx(x)

print('Solution is x = ', x)

np.set_printoptions(precision=4)

