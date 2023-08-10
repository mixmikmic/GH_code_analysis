name = '2016-02-12-jupyter-grace-update'
title = 'Update on using Jupyter Notebook on Grace'
tags = 'jupyter, hpc, anaconda'
author = 'Denis Sergeev'

from nb_tools import connect_notebook_to_post
from IPython.core.display import HTML

html = connect_notebook_to_post(name, title, tags, author)

HTML(html)

