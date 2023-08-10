get_ipython().magic('load_ext autoreload')

get_ipython().magic('autoreload 2')

from Wangview.Display import Display

w = Display('../Wangscape/Wangscape/example3/output')
w.run()

