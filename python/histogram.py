get_ipython().magic('matplotlib inline')
import time
print('Last updated: %s' %time.strftime('%d/%m/%Y'))
import sys
sys.path.insert(0,'../..')
from IPython.display import HTML
from helpers import show_hide
HTML(show_hide)

ls



