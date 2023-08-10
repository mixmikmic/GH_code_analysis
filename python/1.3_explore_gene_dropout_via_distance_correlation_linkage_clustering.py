# Alphabetical order for nonstandard python modules is conventional
# We're doing "import superlongname as abbrev" for our laziness -- 
# -- this way we don't have to type out the whole thing each time.

# Python plotting library
import matplotlib.pyplot as plt

# Dataframes in Python
import pandas as pd

# Statistical plotting library we'll use
import seaborn as sns
# Use the visual style of plots that I prefer and use the 
# "notebook" context, which sets the default font and figure sizes
sns.set(style='whitegrid')

# This is necessary to show the plotted figures inside the notebook -- "inline" with the notebook cells
get_ipython().magic('matplotlib inline')

# Import figure code for interactive widgets
import fig_code

# Read the file - notice it is a URL. pandas can read either URLs or files on your computer
anscombe = pd.read_csv("https://github.com/mwaskom/seaborn-data/raw/master/anscombe.csv")

# Say the variable name with no arguments to look at the data
anscombe

# Make a "grid" of plots based on the column name "dataset"
g = sns.FacetGrid(anscombe, col='dataset')

# Make a regression plot (regplot) using 'x' for the x-axis and 'y' for the y-axis
g.map(sns.regplot, 'x', 'y')

fig_code.interact_anscombe()

for cluster_id, name in fig_code.cluster_id_to_name.items():
    # The 'f' before the string means it's a "format string,"
    # which means it will read the variable names that exist
    # in your workspace. This is a very helpful and convient thing that was
    # just released in Python 3.6! (not available in Python 3.5)
    print('---')
    print(f'{cluster_id}: {name}')

fig_code.plot_color_legend()

fig_code.plot_dropout_interactive()

