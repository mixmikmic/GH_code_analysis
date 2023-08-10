directory = "F:\\PA_UC\\"
stub = 5

get_ipython().magic('run ./ImportData.ipynb')

CROP_SIZE = 80
get_ipython().magic('run ./ImageStitching.ipynb')
display.Image(filename=FileImage)

EDX_save = False
EDX_channels = 2086
get_ipython().magic('run ./EDXProcess.ipynb')

get_ipython().magic('run ./ParticleClassification.ipynb')

dist_required = 75
get_ipython().magic('run ./InterparticleDistance.ipynb')

DISABLE_BOKEH = False
get_ipython().magic('run ./BokehGraphs.ipynb')



