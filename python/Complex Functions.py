get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import *
from scipy import special
import matplotlib

def plot_complex(f, xbounds, ybounds, res=401):

    '''Plot the complex function f.
    INPUTS:
    f - A function handle. Should represent a function from C to C.
    xbounds - A tuple (xmin, xmax) describing the bounds on the real part of the domain.
    ybounds - A tuple (ymin, ymax) describing the bounds on the imaginary part of the domain.
    res - A scalar that determines the resolution of the plot (number of points
        per side). Defaults to 401.
    
    OUTPUTS:
    graph of the function f(z)
    '''
    x = np.linspace(xbounds[0], xbounds[1], res)
    y = np.linspace(ybounds[0], ybounds[1], res)
    X, Y = np.meshgrid(x, y)
    Z = X + 1J*Y
    nums = f(Z)
    plt.pcolormesh(X, Y, np.angle(nums), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.show()

plot_complex(lambda x: np.sqrt(x**2 + 1), (-3, 3), (-3, 3), 401)

def problem2():
    '''Create the plots specified in the problem statement.
    >>>>>>>Please title each of your plots!!!!<<<<<<<
    Print out the answers to the questions.
    '''
    f1 = lambda x: x**2
    f2 = lambda x: x**3
    f3 = lambda x: x**4
    plt.title(r'$x^2$')
    plot_complex(f1, (-5, 5), (-5, 5),res=401)
    plt.title(r'$x^3$')
    plot_complex(f2, (-5, 5), (-5, 5),res=401)
    plt.title(r'$x^4$')
    plot_complex(f3, (-5, 5), (-5, 5),res=401)

problem2()

f4 = lambda x: x**3 - 1j*x**4 - 3*x**6
plt.figure(figsize=(6,6))
plot_complex(f4, (-1, 1), (-1, 1),res=2000)

def problem3():
    '''Create the plots specified in the problem statement.
    Print out the answers to the questions.
    '''
    f5 = lambda x: 1 / x
    f6 = lambda x: x
    plt.title("1/z")
    plot_complex(f5, (-1, 1), (-1, 1),res=1000)
    plt.title("z")
    plot_complex(f6, (-1, 1), (-1, 1),res=1000)

problem3()

f7 = lambda x: x**(-2)
f8 = lambda x: x**(-3)
f9 = lambda x: x**2 + 1j*x**(-1) + x**(-3)
plot_complex(f7, (-1, 1), (-1, 1),res=1000)
plot_complex(f8, (-1, 1), (-1, 1),res=1000)
plot_complex(f9, (-1, 1), (-1, 1),res=1000)

def problem4():
    '''For each plot, create the graph using plot_complex and print out the
    number and order of poles and zeros below it.'''
    plt.title(r'$e^z$')
    f10 = lambda x: np.exp(x)
    plot_complex(f10, (-8, 8), (-8, 8),res=1000)
    plt.title(r'$tan(x)$')
    f11 = lambda x: np.tan(x)
    plot_complex(f11, (-8, 8), (-8, 8),res=1000)
    plt.title(r'the big one')
    f12 = lambda x: ((16*x**4) + (32*x**3) + (32*x**2) + (16*x) + 4) / ((16*x**4)-(16*x**3)+(5*x**2))
    plot_complex(f12, (-1, 1), (-1, 1),res=1000)
    

problem4()

def problem5():
    '''
    For each polynomial, print out each zero and its multiplicity.
    Organize this so the output makes sense.
    '''
    plt.figure(figsize=(10,10))
    f13 = lambda x: -2*x**7 + 2*x**6 - 4*x**5 + 2*x**4 - 2*x**3 - 4*x**2 - 4*x + 4
    plot_complex(f13, (-2, 2), (-2, 2),res=1000)
    plt.figure(figsize=(10,10))
    f14 = lambda x: x**7 + 6*x**6 - 131*x**5 - 419*x**4 + 4906*x**3 - 131*x**2 - 420*x + 4900
    plot_complex(f14, (-15, 15), (-15, 15),res=1000)

problem5()

def problem6():
    '''Create the plots specified in the problem statement.
    Print out the answers to the questions.
    '''
    # part 1
    plt.figure(figsize=(5,5))
    f15 = lambda x: np.sin(1/(100*x))
    plot_complex(f15, (-1, 1), (-1, 1),res=1000)
    plt.figure(figsize=(5,5))
    plot_complex(f15, (-.01, .01), (-.01, .01),res=1000)
    
    # part 2
    f16 = lambda x: x + 1000*x**2
    plt.figure(figsize=(8,8))
    plot_complex(f16, (-1, 1), (-1, 1),res=1000)
    
    plt.figure(figsize=(8,8))
    plot_complex(f16, (-.002, .002), (-.002, .002),res=401)
    
    

problem6()

def problem7():
    '''Create the plots specified in the problem statement.
    Print out the answers to the questions.
    '''
    f17 = lambda x: np.sqrt(x)
    plot_complex(f17, (-1, 1), (-1, 1),res=1000)
    f18 = lambda x: -np.sqrt(x)
    plot_complex(f18, (-1, 1), (-1, 1),res=1000)   
    # it only has half the rainbow (# of angles) because it is a fraction

problem7()

def extraCredit():
    '''
    Create a really awesome complex plot. You can do whatever you want, as long as 
    it's cool and you came up with it on your own.
    You can also animate one of the plots in the lab (look up matplotlib animation)
    Title your plot or print out an explanation of what it is.
    '''
    colors = matplotlib.colors.Normalize(vmin=-np.pi,vmax=np.pi)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #f = lambda x: np.exp(1/x)
    f = lambda x: x**4
    x = np.linspace(-.5,.5,1000)
    y = np.linspace(-.5,.5,1000)
    X, Y = np.meshgrid(x,y)
    Z = X + 1j*Y
    Z = f(Z)
    fc = np.angle(Z)
    fc = (fc+np.pi)/(2*np.pi)
    Z = np.abs(Z)
    Z[Z>100000] = 100000
    surf = ax.plot_surface(X,Y,Z,facecolors=plt.cm.hsv(fc),cmap='hsv',norm=colors)
    plt.title(r'3-D Graph of $e^4$')
    plt.show()

extraCredit()



