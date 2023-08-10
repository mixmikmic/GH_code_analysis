import graphlab

# Load data
# Dataset name : CIFAR 10 dataset (pretty famous dataset)
# this is pre-splitted image data(into training and test data)
image_train = graphlab.SFrame('image_train_data')
image_test = graphlab.SFrame('image_test_data')
graphlab.canvas.set_target('ipynb')

len(image_train)

len(image_test)

image_train.head()
# image_array is pixel data

image_train["image"].show()
# small thumbnail sized images

raw_pixel_model = graphlab.logistic_classifier.create(image_train, target="label", features=["image_array"])

image_test[0:3]['image'].show()
# sample 3 images from "test data"

# these are the labels as per dataset
image_test[0:3]["label"]
# lets see if our model predicts it right

raw_pixel_model.predict(image_test[0:3])
# predicted results are wrong, obviously

raw_pixel_model.evaluate(image_test)
# accuracy is 48% which is very terrible
# hence just using "raw_image_pixels" aint enough

# imagenet has like 1.2million images labelled
# the "deep_features" is derived from the model trained on imagenet dataset
deep_features_model = graphlab.logistic_classifier.create(image_test, features=["deep_features"], target="label")

# sample first 3 images
image_test[0:3]["image"].show()

# their labels
image_test[0:3]["label"]

deep_features_model.predict(image_test[0:3])
# all of the predictions are right, as per above

# lets evaluate accuracy now
deep_features_model.evaluate(image_test)
# accuracy is 82% which is pretty good



