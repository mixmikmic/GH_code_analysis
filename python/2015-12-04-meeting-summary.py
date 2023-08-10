name = '2015-12-04-meeting-summary'
title = 'Python 3, miscellaneous ideas'
tags = 'py3k, version control'
author = 'Matthew Bone'

from nb_tools import connect_notebook_to_post
from IPython.core.display import HTML

html = connect_notebook_to_post(name, title, tags, author)

HTML(html)

