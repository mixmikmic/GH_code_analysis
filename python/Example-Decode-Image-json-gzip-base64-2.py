reset -fs

import gzip
import io
import json
import PIL.Image as img
import numpy as np
import base64
from skimage import color, exposure, transform

json_file = '/Users/carles/Downloads/kadaif%2Fdatasets%2Frecipes%2Frecipes-crawls%2Fimages%2Fepicurious%2F2016-10-07-02-43-25%2F00020eb0e8ebaa40150fa49568d922da.json.gz'

with gzip.open(json_file, 'r') as f:
    d = json.loads(f.read())
    image = img.open(io.BytesIO(base64.b64decode(d['images'][0])))

image

IMG_SIZE = 250

def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img,-1)

    return img

preprocess_img(image)



