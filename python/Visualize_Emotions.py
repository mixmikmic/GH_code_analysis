import os
import sys
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
#widgets with notebook don't work as nicely
module_path = os.path.abspath(os.path.join('../code'))
if module_path not in sys.path:
    sys.path.append(module_path)
from load_plotline import LoadPlotLine, ExploreData

exploration = ExploreData()
select_widget, checkbox_list = exploration.explore()



