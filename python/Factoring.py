from ipywidgets import interact
from IPython.display import display

from sympy import Symbol, Eq, factor, init_printing
init_printing(use_latex='mathjax')

x = Symbol('x')

def factorit(n):
    display(Eq(x**n-1, factor(x**n-1)))

factorit(12)

interact(factorit, n=(2,40));

