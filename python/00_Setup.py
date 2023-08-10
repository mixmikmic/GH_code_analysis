import conx as cx

cx.download("https://github.com/Calysto/conx-notebooks/archive/master.zip")

get_ipython().system(' jupyter trust conx-notebooks-master/*.ipynb')

