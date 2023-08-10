from IPython.display import HTML, display
from helpers import chapters_to_html

HTML(chapters_to_html('./LABS/'))



from IPython.core.display import HTML
def css_styling():
    styles = open("./styles/custom.css", "r").read()
    return HTML(styles)
css_styling()

get_ipython().magic('matplotlib inline')
import time
print('Last updated: %s' %time.strftime('%d/%m/%Y'))
import sys
sys.path.insert(0,'..')
from IPython.display import HTML
from helpers import show_hide
HTML(show_hide)



