9*9

def f(x):
    print(x * x)

f(9)

from ipywidgets import *
from traitlets import link, observe

interact(f, x=(0, 100));

slider = FloatSlider(
    value=7.5,
    min=5.0,
    max=10.0,
    step=0.1,
    description='Input:',
)

slider

slider

slider.value

slider.value = 8

square = slider.value * slider.value
def handle_change(change):
    global square
    square = change.new * change.new
slider.observe(handle_change, 'value')

square

text = FloatText(description='Value')
link((slider, 'value'), (text, 'value'))
VBox([slider, text])







