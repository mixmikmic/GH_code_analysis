import analysis as an
import connectivity as cn

import networkx as nx
import numpy as np

import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
plotly.offline.init_notebook_mode()

# If you want it to save to .html file
# figure = generate_region_graph('Fear199', 'Fear199_regions.csv', 'Fear199_region.html')

figure, count_dict = an.generate_region_graph('Fear199', 'Fear199_regions.csv')

# plot
# iplot(figure)



