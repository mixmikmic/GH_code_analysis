from PIL import Image
from time import time

rand_seed = int(time())
def rng():
    """Return a pseudo-random number between 0 to 8388593.""" 
    global rand_seed
    # For these choice of numbers, see P L'Ecuyer, 
    # "Tables of linear congruential generators of different sizes 
    # and good lattice structure"
    rand_seed = (rand_seed * 653276) % 8388593
    return rand_seed

rgb = bytearray()
for i in range(512*384):
    if i%(128*96)==0:
        # Use similar seeds
        rand_seed = int(time())
    gray = rng()%256
    rgb.extend([gray,gray,gray])

image = Image.frombytes('RGB', (512,384), bytes(rgb))
image.save("/home/xilinx/jupyter_notebooks/examples/data/random_1.jpg", 
           'JPEG')
image

from PIL import Image
from random import randint

rgb = bytearray()
for i in range(512*384):
    gray = randint(0, 255)
    rgb.extend([gray,gray,gray])

image = Image.frombytes('RGB', (512,384), bytes(rgb))
image.save("/home/xilinx/jupyter_notebooks/examples/data/random_2.jpg", 
           'JPEG')
image



