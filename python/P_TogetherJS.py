get_ipython().run_cell_magic('writefile', 'together2.html', '<!DOCTYPE html>\n<html>\n  <meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>\n\n  <body>\n\n<p>Click to draw. Click on the buttons to change colours, erase, resize, or clear. </p>\n<p>Click "Get Together" to begin collaborating with others!</p>\n\n<iframe width="100%" height="500" src="//jsfiddle.net/5f8FL/1/embedded/result/" allowfullscreen="allowfullscreen" frameborder="0"></iframe>\n\n\n  </body>\n</html>')

from IPython.display import IFrame
import re

IFrame('files/together2.html',800,500)





