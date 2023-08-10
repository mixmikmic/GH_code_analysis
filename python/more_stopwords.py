# This code is required so we can display the visualisation
import pyLDAvis
from IPython.core.display import display, HTML

# Changing the cell widths
display(HTML("<style>.container { width:100% !important; }</style>"))

# Setting the max number of rows
pd.options.display.max_rows = 30
# Setting the max number of columns
pd.options.display.max_columns = 50                                         

pyLDAvis.enable_notebook()

from IPython.display import HTML
HTML(filename='vis.html')

