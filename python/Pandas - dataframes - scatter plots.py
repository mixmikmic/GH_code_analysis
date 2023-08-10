import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

get_ipython().magic('matplotlib inline')
pd.set_option('display.notebook_repr_html', False)

x = np.random.random_integers(1, 10, size=40)
y = np.random.random_integers(1, 10, size=40)
size = np.random.random_integers(1, 50, size=40)**2

x, y, size

df = pd.DataFrame({'kolom1': x, 'kolom2': y, 'kolom3' : size})

plt.scatter(df.kolom1, df.kolom2, df.kolom3, alpha=0.5)
plt.grid()
plt.title('Matplotlib Scatter')

df.plot(kind='scatter', x='kolom1', y='kolom2', s=df.kolom3, title='Pandas Scatter', alpha=0.5)

