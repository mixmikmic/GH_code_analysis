import os
# Here you need to tell the notebook where your IDV executable is, via an environment variable. 
# Mac default is /Applications/IDV_version
# Windows default is perhaps similar -- find it please, Jimmy! 

os.environ['IDV_HOME']='/Applications/IDV_5.4'

get_ipython().run_line_magic('reload_ext', 'drilsdown')

# Capture the initial image 
get_ipython().run_line_magic('makeImage', '-caption Initial_Image')

#animated gif
get_ipython().run_line_magic('makeMovie', "-caption 'Animation of whole time sequence'")

get_ipython().run_line_magic('makeImage', '-caption Baja_closeup')

get_ipython().run_line_magic('makeImage', '-caption Atlantic_closeup')

get_ipython().run_line_magic('makeImage', '-caption Alaska')

get_ipython().run_line_magic('makeImage', '-caption Hudsons_Bay')

from drilsdown import Idv
Idv.dataUrl

# Importing an animated .gif

from IPython.display import Image
from IPython.display import display
with open('/Users/bem/Downloads/latest72hrs_global.gif','rb') as f:
    display(Image(f.read()), format='png')

# Importing an image

from IPython.display import Image
Image("/Users/bem/Dropbox/AAA_ipython_notebooks/DRILSDOWN_DEMO/LMT_METEO_TEXTBOOK/images/Initial_Image.png")





get_ipython().run_line_magic('publishNotebook', '')

get_ipython().run_line_magic('createCaseStudy', '')



