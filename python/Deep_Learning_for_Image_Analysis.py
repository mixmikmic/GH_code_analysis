import graphlab 
import graphlab.mxnet
graphlab.canvas.set_target('ipynb')

products_train = graphlab.SFrame('products_train.sf/')
products_test = graphlab.SFrame('products_test.sf/')

products_test['image'].show()

pretrained_model = graphlab.mxnet.pretrained_model.load_path('mxnet_models/imagenet1k_inception_bn/')

predictions = pretrained_model.predict_topk(products_test.head(10), k=1)

predictions['label']

#products_train['extracted_features'] = pretrained_model.extract_features(products_train)

#products_test['extracted_features'] = pretrained_model.extract_features(products_test)

transfer_model = graphlab.logistic_classifier.create(products_train, features=['extracted_features'], target='label', validation_set=products_test)

products_all = products_test.append(products_train)
products_all = products_all.add_row_number()

nearest_neighbors_model = graphlab.nearest_neighbors.create(products_all, features=['extracted_features'])

query = products_all[0:1]
query['image'].show()

query_results = nearest_neighbors_model.query(query)

query_results

filtered_results = products_all.filter_by(query_results['reference_label'], 'id')

filtered_results['image'].show()

network = graphlab.deeplearning.create(products_train, target='label')


network 

products_test['image'] = graphlab.image_analysis.resize(products_test['image'], 224, 224, 3)

neural_net_model = graphlab.neuralnet_classifier.create(products_train,network = network, features=['image'], target='label', validation_set=products_test, max_iterations = 3)

network

network.layers[3] = graphlab.deeplearning.layers.FullConnectionLayer(200)

network

neural_net_model = graphlab.neuralnet_classifier.create(products_train,network = network, features=['image'], target='label', validation_set=products_test, max_iterations = 3)

detector_query = graphlab.image_analysis.load_images('detection_query')

detector_query['image'][0].show()

detector = graphlab.mxnet.pretrained_model.load_path('mxnet_models/coco_vgg_16/')

detections = detector.detect(detector_query['image'][0])

detections

backpack_detections = detections.filter_by(['backpack'], 'class')

backpack_detections

visualize = detector.visualize_detection(detector_query['image'][0], backpack_detections)

visualize.show()

def crop(gl_img, box):    
    _format = {'JPG': 0, 'PNG': 1, 'RAW': 2, 'UNDEFINED': 3}
    pil_img = gl_img._to_pil_image()
    cropped = pil_img.crop([int(c) for c in box])
    
    height = cropped.size[1]
    width = cropped.size[0]
    if cropped.mode == 'L':
        image_data = bytearray([z for z in cropped.getdata()])
        channels = 1
    elif cropped.mode == 'RGB':
        image_data = bytearray([z for l in cropped.getdata() for z in l ])
        channels = 3
    else:
        image_data = bytearray([z for l in cropped.getdata() for z in l])
        channels = 4
    format_enum = _format['RAW']
    image_data_size = len(image_data)

    img = graphlab.Image(_image_data=image_data, _width=width, _height=height, _channels=channels, _format_enum=format_enum, _image_data_size=image_data_size)
    return img

cropped = crop(detector_query['image'][0], backpack_detections['box'][0])

cropped.show()

query_sf = graphlab.SFrame({'image' : [cropped]})

query_sf['image'].show()

query_sf['extracted_features'] = pretrained_model.extract_feature(query_sf)

query_sf

products_all

similar_backpacks = nearest_neighbors_model.query(query_sf)

similar_backpacks

filtered_similar_backpacks = products_all.filter_by(similar_backpacks['reference_label'], 'id')

filtered_similar_backpacks['image'].show()



