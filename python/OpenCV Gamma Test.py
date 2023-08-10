import os
import tempfile

import cv2
import IPython.display
import numpy as np

def show_cv_image(img):
    # Can not use Matplotlib because it scales images again.
    (handle, tmpname) = tempfile.mkstemp(suffix='.png')
    os.close(handle)
    cv2.imwrite(tmpname, img)
    IPython.display.display(IPython.display.Image(tmpname))
    os.unlink(tmpname)

img = cv2.imread('gamma test image.png')
show_cv_image(img)

res = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
show_cv_image(res)

img2 = cv2.imread('gamma_dalai_lama_gray.jpg')
print('Original image')
show_cv_image(img2)
print('Resized image')
res2 = cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
show_cv_image(res2)

from srgb import to_linear, from_linear

fimg = to_linear(img)
fres = cv2.resize(fimg, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
fres = from_linear(fres)
show_cv_image(fres)

fimg2 = to_linear(img2)
fres2 = cv2.resize(fimg2, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
fres2 = from_linear(fres2)
show_cv_image(fres2)



