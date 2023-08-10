import graphlab

# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)

image_train = graphlab.SFrame('image_train_data/')

# deep_learning_model = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')
# image_train['deep_features'] = deep_learning_model.extract_features(image_train)

image_train.head()

knn_model = graphlab.nearest_neighbors.create(image_train,features=['deep_features'],label='id')

graphlab.canvas.set_target('ipynb')
cat = image_train[18:19]
cat['image'].show()

knn_model.query(cat)

def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'],'id')

cat_neighbors = get_images_from_ids(knn_model.query(cat))

cat_neighbors['image'].show()

car = image_train[8:9]
car['image'].show()

get_images_from_ids(knn_model.query(car))['image'].show()

show_neighbors = lambda i: get_images_from_ids(knn_model.query(image_train[i:i+1]))['image'].show()

show_neighbors(8)

show_neighbors(26)

automobile_data = image_train[image_train['label'] == 'automobile']
cat_data = image_train[image_train['label'] == 'cat']
dog_data = image_train[image_train['label'] == 'dog']
bird_data = image_train[image_train['label'] == 'bird']

automobile_model = graphlab.nearest_neighbors.create(automobile_data,features=['deep_features'],label='id')
cat_model = graphlab.nearest_neighbors.create(cat_data,features=['deep_features'],label='id')
dog_model = graphlab.nearest_neighbors.create(dog_data,features=['deep_features'],label='id')
bird_model = graphlab.nearest_neighbors.create(bird_data,features=['deep_features'],label='id')

image_test = graphlab.SFrame('image_test_data/')

image_test[0:1]['image'].show()

