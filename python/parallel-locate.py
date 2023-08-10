from ipyparallel import Client
client = Client()
view = client.load_balanced_view()

client[:]

get_ipython().run_cell_magic('px', '', 'import trackpy as tp')

import pims
import trackpy as tp

def gray(image):
    return image[:, :, 0]

frames = pims.ImageSequence('../sample_data/bulk_water/*.png', process_func=gray)

curried_locate = lambda image: tp.locate(image, 13, invert=True)

view.map(curried_locate, frames[:4])  # Optionally, prime each engine: make it set up FFTW.

get_ipython().run_cell_magic('timeit', '', 'amr = view.map_async(curried_locate, frames[:32])\namr.wait_interactive()\nresults = amr.get()')

get_ipython().run_cell_magic('timeit', '', 'serial_result = list(map(curried_locate, frames[:32]))')

