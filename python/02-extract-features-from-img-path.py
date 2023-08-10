src_path = '/jupyter-nfs/home/myazdani/feature-weights/data/images/data/VOC2007/'
write_path = "../../VOC2007-ResNet50-pool5.csv"

import os

image_types = (".jpg", ".png", ".JPG", ".PNG", ".JPEG", ".jpeg",".tif", ".tiff", ".TIFF", '.TIF')
 
image_paths = []  
for root, dirs, files in os.walk(src_path):
    image_paths.extend([os.path.join(root, f) for f in files if f.endswith(image_types)])
    
print 'number of images is', len(image_paths)

from skicaffe import SkiCaffe

caffe_root = '/usr/local/caffe/'
model_file = './models/ResNet-50-model.caffemodel'
prototxt_file = './models/ResNet-50-deploy.prototxt'

ResNet = SkiCaffe(caffe_root = caffe_root,
                  model_prototxt_path = prototxt_file, 
                  model_trained_path = model_file, 
                  include_labels = False,
                  include_image_paths = True,
                  return_type = "pandasDF")

ResNet.fit()
print 'Number of layers:', len(ResNet.layer_sizes)
ResNet.layer_sizes

image_features = ResNet.transform(X = image_paths, layer_name='pool5')

image_features.head()

image_features.to_csv(write_path, index = False)

