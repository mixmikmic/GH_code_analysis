import graphlab 
images = graphlab.SFrame('https://static.turi.com/datasets/caltech_101/caltech_101_images')

# Only do this if you have a GPU
#pretrained_model = graphlab.load_model('https://static.turi.com/models/imagenet_model_iter45')
#images['extracted_features'] = pretrained_model.extract_features(images)

# If you do not have a GPU, do this instead. 
images['extracted_features'] = graphlab.SArray('https://static.turi.com/models/pre_extracted_features.gl')

images

nearest_neighbor_model = graphlab.nearest_neighbors.create(images, features=['extracted_features'])

similar_images = nearest_neighbor_model.query(images, k = 2)

similar_images

similar_images = similar_images[similar_images['query_label'] != similar_images['reference_label']]

similar_images

graphlab.canvas.set_target('ipynb')
graphlab.SArray([images['image'][9]]).show()

graphlab.SArray([images['image'][1710]]).show()

graphlab.SArray([images['image'][0]]).show()

graphlab.SArray([images['image'][1535]]).show()

