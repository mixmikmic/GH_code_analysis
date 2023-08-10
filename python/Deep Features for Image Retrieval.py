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

bird = image_train[image_train['label'] == 'bird']
dog = image_train[image_train['label'] == 'dog']
cat = image_train[image_train['label'] == 'cat']
automobile = image_train[image_train['label'] == 'automobile']

bird_model = graphlab.nearest_neighbors.create(bird,features=['deep_features'],
                                             label='id')
dog_model = graphlab.nearest_neighbors.create(dog,features=['deep_features'],
                                             label='id')
cat_model = graphlab.nearest_neighbors.create(cat,features=['deep_features'],
                                             label='id')
automobile_model = graphlab.nearest_neighbors.create(automobile,features=['deep_features'],
                                             label='id')

image_test = graphlab.SFrame('image_test_data/')

graphlab.canvas.set_target('ipynb')
image_test[0:1]['image'].show()

query_result = cat_model.query(image_test[0:1])


cat.filter_by(query_result['reference_label'],'id')['image'].show()

query_result = dog_model.query(image_test[0:1])
dog.filter_by(query_result['reference_label'],'id')['image'].show()

cat_model.query(image_test[0:1])['distance'].mean()

dog_model.query(image_test[0:1])['distance'].mean()

image_test_bird = image_test[image_test['label'] == 'bird']
image_test_dog = image_test[image_test['label'] == 'dog']
image_test_cat = image_test[image_test['label'] == 'cat']
image_test_automobile = image_test[image_test['label'] == 'automobile']

dog_cat_neighbors = cat_model.query(image_test_dog, k=1)

dog_distances = graphlab.SFrame({'dog-dog': dog_model.query(image_test_dog, k=1)['distance'],
                                 'dog-cat': cat_model.query(image_test_dog, k=1)['distance'],
                                 'dog-automobile': automobile_model.query(image_test_dog, k=1)['distance'],
                                 'dog-bird': bird_model.query(image_test_dog, k=1)['distance'],
                                })

dog_distances

def is_dog_correct(row):
    return 1 if row['dog-dog'] < row[min(['dog-automobile','dog-bird','dog-cat'],key = lambda key: row[key])] else 0

dog_distances.apply(is_dog_correct).sum()

len(dog_distances)

cat_distances = graphlab.SFrame({'cat-dog': dog_model.query(image_test_cat, k=1)['distance'],
                                 'cat-cat': cat_model.query(image_test_cat, k=1)['distance'],
                                 'cat-automobile': automobile_model.query(image_test_cat, k=1)['distance'],
                                 'cat-bird': bird_model.query(image_test_cat, k=1)['distance'],
                                })

def is_cat_correct(row):
    return 1 if row['cat-cat'] < row[min(['cat-automobile','cat-bird','cat-dog'],key = lambda key: row[key])] else 0

cat_distances.apply(is_cat_correct).sum()

