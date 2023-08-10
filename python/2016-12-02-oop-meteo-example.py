name = '2016-12-02-oop-meteo-example'
title = 'Object-Oriented Programming in meteorological data analysis'
tags = 'oop, iris'
author = 'Adrian Matthews'

from nb_tools import connect_notebook_to_post
from IPython.core.display import HTML, Image

html = connect_notebook_to_post(name, title, tags, author)

HTML(html)

