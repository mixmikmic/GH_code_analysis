import graphlab

image_train = graphlab.SFrame('image_train_data/')

#deep_learning_model = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')
#image_train['deep_features'] = deep_learning_model.extract_features(image_train)

image_train.head()

knn_model = graphlab.nearest_neighbors.create(image_train,features=['deep_features'],
                                             label='id')

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

show_neighbors(1222)

show_neighbors(2000)

image_train['label'].sketch_summary()

dog_data=image_train[image_train['label'] =='dog'] 

cat_data=image_train[image_train['label'] =='cat'] 

auto_data=image_train[image_train['label'] =='automobile'] 

bird_data=image_train[image_train['label'] =='bird'] 

dog_model = graphlab.nearest_neighbors.create(dog_data,features=['deep_features'],
                                             label='id')

cat_model = graphlab.nearest_neighbors.create(cat_data,features=['deep_features'],
                                             label='id')

auto_model = graphlab.nearest_neighbors.create(auto_data,features=['deep_features'],
                                             label='id')

bird_model = graphlab.nearest_neighbors.create(bird_data,features=['deep_features'],
                                             label='id')

image_test[0:1]

graphlab.canvas.set_target('ipynb')
cat_test = image_test[0:1]
cat_test['image'].show()

def get_images_cat_ids(query_result):
    return cat_data.filter_by(query_result['reference_label'],'id')

cat_nearest = get_images_cat_ids(cat_model.query(cat_test))

cat_nearest['image'].show()

def get_images_dog_ids(query_result):
    return dog_data.filter_by(query_result['reference_label'],'id')

dog_nearest = get_images_dog_ids(dog_model.query(cat_test))

dog_nearest['image'].show()

cat_model.query(cat_test)

cat_model.query(cat_test)['distance'].mean()

dog_model.query(cat_test)

dog_model.query(cat_test)['distance'].mean()

image_test_dog=image_test[image_test['label'] =='dog'] 
image_test_cat=image_test[image_test['label'] =='cat'] 
image_test_auto=image_test[image_test['label'] =='automobile'] 
image_test_bird=image_test[image_test['label'] =='bird'] 

graphlab.canvas.set_target('ipynb')
dog_test = image_test_dog
cat_test = image_test_cat
auto_test = image_test_auto
bird_test = image_test_bird

dog_model.query(dog_test)
cat_model.query(cat_test)
auto_model.query(auto_test)
bird_model.query(bird_test)

dog_cat_neighbors = cat_model.query(image_test_dog, k=1)



