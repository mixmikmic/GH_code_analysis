# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import nbinteract as nbi

sns.set()
sns.set_context('talk')
pd.options.display.max_rows = 7

# HIDDEN
students = pd.read_csv('roster.csv')
students['Name'] = students['Name'].str.lower()

students

print("There are", len(students), "students on the roster.")

students['Role'].value_counts().to_frame()

sns.distplot(students['Name'].str.len(), rug=True, axlabel="Number of Characters")

