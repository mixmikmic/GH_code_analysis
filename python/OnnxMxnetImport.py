import onnx_mxnet
from IPython.core.display import display
sym, params = onnx_mxnet.import_model('super_resolution.onnx')

import numpy as np
import mxnet as mx
from PIL import Image
img = Image.open('super-res-input.jpg').resize((224, 224))
display(img) # show the image
img_ycbcr = img.convert("YCbCr")
img_y, img_cb, img_cr = img_ycbcr.split()
x = mx.nd.array(np.array(img_y)[np.newaxis, np.newaxis, :, :])

mod = mx.mod.Module(symbol=sym, data_names=['input_0'], label_names=None)
mod.bind(for_training=False, data_shapes=[('input_0',x.shape)])
mod.set_params(arg_params=params, aux_params=None)

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])
mod.forward(Batch([x]))
output = mod.get_outputs()[0][0][0]

img_out_y = Image.fromarray(np.uint8((output.asnumpy().clip(0, 255)), mode='L'))
result_img = Image.merge("YCbCr", [
        	img_out_y,
        	img_cb.resize(img_out_y.size, Image.BICUBIC),
        	img_cr.resize(img_out_y.size, Image.BICUBIC)
]).convert("RGB")
display(result_img)
result_img.save("super_res_output.jpg")

