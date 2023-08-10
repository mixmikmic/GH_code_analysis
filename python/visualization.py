get_ipython().run_line_magic('matplotlib', 'inline')
import numpy
import matplotlib
import sys 
sys.path.append("..")
from handwriting.utils import visualization_utils as vu

strokes = numpy.load('../data/raw/strokes.npy', encoding="latin1")
with open('../data/raw/sentences.txt') as f:
    texts = f.readlines()
print(strokes.shape)
print(strokes[0].shape)

idx = 0
stroke = strokes[idx]
text = texts[idx]
vu.plot_stroke(stroke)
print('TEXT:', text)

stroke



