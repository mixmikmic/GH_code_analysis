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

image_train['label'].sketch_summary()

image_train_auto = image_train[image_train['label'] == 'automobile']

image_train_cat = image_train[image_train['label'] == 'cat']

image_train_dog = image_train[image_train['label'] == 'dog']

image_train_bird = image_train[image_train['label'] == 'bird']

auto_model = graphlab.nearest_neighbors.create(image_train_auto, features = ['deep_features'], label = 'id')

cat_model = graphlab.nearest_neighbors.create(image_train_cat, features = ['deep_features'], label = 'id')

dog_model = graphlab.nearest_neighbors.create(image_train_dog, features = ['deep_features'], label = 'id')

bird_model = graphlab.nearest_neighbors.create(image_train_bird, features = ['deep_features'], label = 'id')

image_test = graphlab.SFrame('image_test_data/')

image_test[0:1].show()

nearest_cats = cat_model.query(image_test[0:1])

nearest_cats

get_images_from_ids(nearest_cats[nearest_cats['reference_label'] == 16289])['image'].show()

nearest_dogs = dog_model.query(image_test[0:1])

nearest_dogs

get_images_from_ids(nearest_dogs[0:1])['image'].show()

nearest_cats['distance'].mean()

nearest_dogs['distance'].mean()

image_test_auto = image_test[image_test['label'] == 'automobile']

image_test_cat = image_test[image_test['label'] == 'cat']

image_test_dog = image_test[image_test['label'] == 'dog']

image_test_bird = image_test[image_test['label'] == 'bird']

dog_auto_neighbors = auto_model.query(image_test_dog, k=1)

dog_cat_neighbors = cat_model.query(image_test_dog, k=1)

dog_dog_neighbors = dog_model.query(image_test_dog, k=1)

dog_bird_neighbors = bird_model.query(image_test_dog, k=1)

dog_distances = graphlab.SFrame({'dog-auto': dog_auto_neighbors['distance'], 'dog-cat' : dog_cat_neighbors['distance'], 
                                 'dog-dog' : dog_dog_neighbors['distance'], 'dog-bird' : dog_bird_neighbors['distance']})

dog_distances

dog_distances['dog-cat'][2]

def is_dog_correct(row):
    d = row['dog-dog']
    if d > row['dog-auto'] or d > row['dog-cat'] or d > row['dog-bird']:
        return 0
    else:
        return 1    

total_correct = dog_distances.apply(is_dog_correct).sum()

total_correct

total_number = len(image_test_dog)

( float(total_correct) * 100 ) / float(total_number)

