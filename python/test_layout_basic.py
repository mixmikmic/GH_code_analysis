from IPython.display import Image

Image('https://pbs.twimg.com/profile_images/351495785/urth_caffe.png_400x400.jpg')

from IPython.display import display
from ipywidgets import FloatText, FloatSlider, FloatProgress
from traitlets import link

a = FloatText()
b = FloatSlider()
c = FloatProgress()
display(a,b,c)

link1 = link((a, 'value'), (b, 'value'))
link2 = link((a, 'value'), (c, 'value'))



