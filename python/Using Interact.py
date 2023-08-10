from __future__ import print_function # for python 2
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

def f(x):
    print(x)

f(10)

interact(f, x=10);

interact(f, x=True);

interact(f, x='Hi there!');

@interact(x=True, y=1.0)
def g(x, y):
    print(x, y)

def h(p, q):
    print(p, q)

h(5, 10)

interact(h, p=5, q=fixed(20));

interact(f, x=widgets.IntSlider(min=-10,max=30,step=1,value=10));

interact(f, x=(0,4));

interact(f, x=(0,8,2));

interact(f, x=(0.0,10.0));

interact(f, x=(0.0,10.0,0.01));

@interact(x=(0.0,20.0,0.5))
def h(x=5.5):
    print(x)

interact(f, x=['apples','oranges']);

interact(f, x={'a': 10, 'b': 20});

def f(a, b):
    return a+b

f(1,2)

w = interactive(f, a=10, b=20)

type(w)

w.children

from IPython.display import display
display(w)

w.kwargs

w.result



