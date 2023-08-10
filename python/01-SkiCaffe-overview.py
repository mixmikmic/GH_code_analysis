from skicaffe import SkiCaffe

caffe_root = '/usr/local/caffe/'
model_file = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
prototxt_file = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'

DLmodel = SkiCaffe(caffe_root = caffe_root,
                   model_prototxt_path = prototxt_file, 
                   model_trained_path = model_file, 
                   include_labels = True,
                   return_type = "pandasDF")

DLmodel.fit()
print 'Number of layers:', len(DLmodel.layer_sizes)
DLmodel.layer_sizes

image_paths = ['./images/cat.jpg', 
               './images/1404329745.jpg']

image_features = DLmodel.transform(X = image_paths)
image_features.head()

DLmodel.include_labels = False
DLmodel.return_type = 'numpy_array'
image_features = DLmodel.transform(X = image_paths, layer_name='fc7')
image_features

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

image_features = ResNet.transform(X = image_paths)
image_features.head()

