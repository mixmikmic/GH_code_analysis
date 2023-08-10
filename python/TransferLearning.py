import os

import numpy as np

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image

# constants
DATA_PATH = '/app/data'
TRAIN_DATA = os.path.join(DATA_PATH, 'train')
VAL_DATA = os.path.join(DATA_PATH, 'val')
IMG_HEIGHT, IMG_WIDTH = 299, 299
BATCH_SIZE = 32
STEPS_PER_EPOCH = 50
VALIDATION_STEPS = 20
EPOCHS = 5

dog_path = os.path.join(TRAIN_DATA, 'dog')
dog_pictures = os.listdir(dog_path)
sampled_dog_image = np.random.choice(dog_pictures)
Image.open(os.path.join(dog_path, sampled_dog_image))

cat_path = os.path.join(TRAIN_DATA, 'cat')
cat_pictures = os.listdir(cat_path)
sampled_cat_image = os.path.join(cat_path, np.random.choice(cat_pictures))
Image.open(sampled_cat_image)

# These parameters control the range of augmentation we'll use.
generator_factory = ImageDataGenerator(
        rotation_range=10, 
        width_shift_range=0.2,
        height_shift_range=0.2,            
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

preview_dir = 'preview'
if not os.path.exists(preview_dir):
    os.mkdir(preview_dir)

img = load_img(sampled_cat_image, target_size=(IMG_HEIGHT, IMG_WIDTH))
img = np.expand_dims(img, axis=0)
# the `flow` method creates a generator that yields batches of images
img_generator = generator_factory.flow(img, batch_size=1, save_to_dir=preview_dir, save_format='jpg')

sampled_transformed_img = next(img_generator)
sampled_transformed_img = np.squeeze(sampled_transformed_img, axis=0)
sampled_transformed_img = array_to_img(sampled_transformed_img)
sampled_transformed_img

def setup_model():
    # Load up the pretrained Inception v3 model.
    img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))  # 3 color channels for red, green, and blue
    inception = InceptionV3(include_top=False, input_tensor=img)
    
    # Freezing the layers
    # For cat/dog classifier, we'll freeze all the layers. If you have more data and/or your problem is very 
    # different from the original task, you should experiment with leaving some of the later layers unfrozen.
    for layer in inception.layers:
        layer.trainable = False
        # Batch norm layers have parameters that get updated regardless of the `trainable` property state. 
        # Setting `momentum` to 1.0 prevents those updates from happening.
        if type(layer) == 'BatchNormalization':
            layer.momentum = 1.0
    
    # The last hidden layer has 8x8 spatial dimensions (downsampled from 299x299) and 2048 channels.
    # We'll average everything spatially to get a 2048D descriptor for the entire image.
    features = GlobalAveragePooling2D()(inception.layers[-1].output)
    
    # This 2048D descriptor is now the features we'll use to train a single layer.
    # More layers can be added here if you have enough data.
    classifier = Dense(2, activation='softmax')(features)
    model = Model(inputs=img, outputs=classifier)
    return model

model = setup_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def setup_generator(data_dir):
    generator_factory = ImageDataGenerator(
            rotation_range=10, 
            width_shift_range=0.2,
            height_shift_range=0.2,                                   
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            preprocessing_function=preprocess_input)
    
    generator = generator_factory.flow_from_directory(
            data_dir, 
            target_size=(IMG_HEIGHT, IMG_WIDTH))
    return generator

train_generator = setup_generator(TRAIN_DATA)
val_generator = setup_generator(VAL_DATA)

train_history = model.fit_generator(
        train_generator, 
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=VALIDATION_STEPS)

def undo_image_preprocessing(img):
    x = img / 2.
    x += 0.5
    x *= 255.
    return x

def sample_image_and_label(generator):
    """Sample image and label

    Returns both the preprocessed image data and the unprocessed image
    to make it easy to view and run predictions."""
    image_batch, label_batch = next(generator)
    test_img_data, test_label = image_batch[0], label_batch[0]

    test_img = undo_image_preprocessing(test_img_data)
    test_img = test_img.astype(np.uint8)
    test_img = Image.fromarray(test_img, mode='RGB')
    
    test_img_data = np.expand_dims(test_img_data, axis=0)

    return test_img_data, test_img, test_label

test_img_data, test_img, test_label = sample_image_and_label(val_generator)

prediction = model.predict(test_img_data)
print 'label:', np.argmax(test_label), 'prediction:', np.argmax(prediction)
print '=' * 22

test_img



