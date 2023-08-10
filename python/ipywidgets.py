from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from IPython.display import display

dd1 = widgets.Dropdown(
    options={'One': 1, 'Two': 2, 'Three': 3},
    value=2,
    description='Number:',
)
dd2 = widgets.Dropdown(
    options={'One': 1, 'Two': 2, 'Three': 3},
    value=2,
    description='Number:',
)
display(dd1, dd2)

def on_dd1_change(change):
    dd2.options = {'new': change['new'], 'old': change['old']}
    dd2.value = change['new']
    #print(change['new'])

dd1.observe(on_dd1_change, names='value')

int_range = widgets.IntSlider()
display(int_range)

def on_value_change(change):
    print(change['new'])

int_range.observe(on_value_change, names='value')

def f(x):
    return x

interact(f, x=(-100,100,5), continuous_update=False)

interact(f, x=(-100,100,5), continuous_update=True)

interact(f, x=(-100,100,5), __manual=True)







