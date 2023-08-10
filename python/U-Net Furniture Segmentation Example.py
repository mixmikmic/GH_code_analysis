get_ipython().run_line_magic('matplotlib', 'inline')
import brine
import cv2
import numpy as np
from model.augmentations import randomHueSaturationValue, randomShiftScaleRotate, randomHorizontalFlip
import model.u_net as unet
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array

get_ipython().system('brine install rohan/chairs-with-masks')

chairs = brine.load_dataset('rohan/chairs-with-masks')

print('Dataset size:', len(chairs))
print('Columns:', chairs.columns)
print(chairs[23])

chairs.load_image(chairs[23].image)

chairs.load_image(chairs[23].mask)

model = unet.get_unet_256(num_classes=1)
model.summary()

SIZE = (256, 256)

# Method to set grayscale mask values to either 0 or 255
def fix_mask(mask):
    mask[mask < 100] = 0.0
    mask[mask >= 100] = 255.0

# Processing function for the training data
def train_process(data):
    img, mask = data
    img = img[:,:,:3]
    mask = mask[:, :, :3]
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    fix_mask(mask)
    img = cv2.resize(img, SIZE)
    mask = cv2.resize(mask, SIZE)
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-50, 50),
                                   sat_shift_limit=(0, 0),
                                   val_shift_limit=(-15, 15))
    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.0625, 0.0625),
                                       scale_limit=(-0.1, 0.1),
                                       rotate_limit=(-20, 20))
    img, mask = randomHorizontalFlip(img, mask)
    fix_mask(mask)
    img = img/255.
    mask = mask/255.
    mask = np.expand_dims(mask, axis=2)
    return (img, mask)

# Processing function for the validation data, no data augmentation
def validation_process(data):
    img, mask = data
    img = img[:,:,:3]
    mask = mask[:, :, :3]
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    fix_mask(mask)
    img = cv2.resize(img, SIZE)
    mask = cv2.resize(mask, SIZE)
    fix_mask(mask)
    img = img/255.
    mask = mask/255.
    mask = np.expand_dims(mask, axis=2)
    return (img, mask)

BATCH_SIZE = 1

validation_fold, train_fold = chairs.create_folds((20,))
print('Validation fold size:', len(validation_fold))
print('Train fold size:', len(train_fold))

train_generator = train_fold.to_keras('image',  # Which column we want to use for our 'xs'
                                      'mask',   # Which column we want to use for our 'ys'
                                      batch_size=BATCH_SIZE,
                                      shuffle=True, 
                                      processing_function=train_process)

validation_generator = validation_fold.to_keras('image',
                                                'mask',
                                                batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                processing_function=validation_process)

image, mask = next(train_generator)
plt.imshow(image[0])

plt.imshow(mask[0].reshape(SIZE))

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True)]

epochs=100
model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_generator.steps_per_epoch(),
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.steps_per_epoch())

model.load_weights('weights/best_weights.hdf5')

def predict_one():
    image_batch, mask_batch = next(validation_generator)
    predicted_mask_batch = model.predict(image_batch)
    image = image_batch[0]
    predicted_mask = predicted_mask_batch[0].reshape(SIZE)
    plt.imshow(image)
    plt.imshow(predicted_mask, alpha=0.6)

predict_one()

