import onnx
from onnx_tf.backend import prepare
model = onnx.load('assets/super_resolution.onnx')
tf_rep = prepare(model)

print(tf_rep.predict_net)
print('-----')
print(tf_rep.input_dict)
print('-----')
print(tf_rep.uninitialized)

import numpy as np
from PIL import Image
img = Image.open('assets/super-res-input.jpg').resize((224, 224))
display(img) # show the image
img_ycbcr = img.convert("YCbCr")
img_y, img_cb, img_cr = img_ycbcr.split()
doggy_y = np.asarray(img_y, dtype=np.float32)[np.newaxis, np.newaxis, :, :]

big_doggy = tf_rep.run(doggy_y)._0
print(big_doggy.shape)

img_out_y = Image.fromarray(np.uint8(big_doggy[0, 0, :, :].clip(0, 255)), mode='L')
result_img = Image.merge("YCbCr", [
    img_out_y,
    img_cb.resize(img_out_y.size, Image.BICUBIC),
    img_cr.resize(img_out_y.size, Image.BICUBIC),
]).convert("RGB")
display(result_img)
result_img.save('output/super_res_output.jpg')



