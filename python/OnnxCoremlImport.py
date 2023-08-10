import onnx
import onnx_coreml
model = onnx.load('assets/super_resolution.onnx')

cml = onnx_coreml.convert(model)
print(type(cml))
cml.save('output/super_resolution.mlmodel')

import numpy as np
from PIL import Image
from IPython.core.display import display

# load the image
img = Image.open("assets/cat.jpg").resize([224, 224])
display(img)

# load the resized image and convert it to Ybr format
img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

# layout should be CxHxW - CoreML supports either rank-3 or rank-5 tensors
data = np.array(img_y)[np.newaxis, :, :].astype(np.float32)
print(data.shape)

# name of input and output of our model
input_name = cml.get_spec().description.input[0].name
output_name = cml.get_spec().description.output[0].name

img_out = cml.predict({input_name: data})[output_name]

img_out_y = Image.fromarray(np.uint8((img_out[0,0]).clip(0, 255)), mode='L')

# get the output image follow post-processing step from PyTorch implementation
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")

# Show the picture
display(final_img)

