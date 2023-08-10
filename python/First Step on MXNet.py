import mxnet as mx
import numpy as np

#load validation data
data_shape = (3, 224, 224)
val = mx.io.ImageRecordIter(
        path_imgrec = "imagenet_val.rec",
        mean_r      = 123.68,
        mean_g      = 116.779,
        mean_b      = 103.939,
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = 1)

#load pre-trained vgg-16 model
model_loaded = mx.model.FeedForward.load('vgg-16/vgg_16_deploy', 1)

y = model_loaded.predict(val)

labels = open('ILSVRC2012_validation_ground_truth.txt', 'r')
index = 1
correct = 0
for line in labels.readlines():
    label = int(line)
    predict = np.argsort(y[index-1])[::-1][:5]
    if label in predict:
        correct += 1
    if index == len(y):
        break
    index += 1
print float(correct) / len(y)



