from ipywidgets import interact
from IPython.display import display
import pandas as pd
import numpy as np

@interact(count=['5','10','15'])
def render(count):
    count = int(count)
    data = np.random.randn(count, count)
    display(pd.DataFrame(data))

