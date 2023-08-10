cd /Users/Will/Documents/GITHUB/class_project/class_project/HW7

from data_cleaning_utils import *
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

cd /Users/Will/Documents/GITHUB/class_project/class_project/Data/Amazon

raw = import_data()

raw.head()

smooth = smooth_data("pH")

reduced = reducer()

reduced.head()

