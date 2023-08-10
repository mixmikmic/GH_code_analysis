from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input

TRAIN_DIR = "mias_mini-train"
TEST_DIR = "mias_mini-test"
IM_WIDTH, IM_HEIGHT = 299, 299
FC_SIZE = 1024
batch_size = 75
NUM_CLASSES = 3
NUM_EPOCHS = 100

train_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
)

train_generator

import os
import numpy as np
import cv2


images = []
classes = ["0","1",'2']

for cat in classes:
    TRAIN_DIR = "mias_mini-train" + '/' + cat
    TEST_DIR = "mias_mini-test" +'/' + cat

    for imgName in os.listdir(TRAIN_DIR):
        if "png" in imgName:
            images.append(cv2.imread(TRAIN_DIR + '/' + imgName))

X_train = np.array(images[1:])  

X_train.shape



get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

class DCGAN():
    def __init__(self):
        self.img_rows = 256 
        self.img_cols = 256
        self.channels = 3

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        #noise_shape = (self.img_rows, self.img_cols, self.channels)
        #z = Input(shape=noise_shape)
        #z = Input((self.channels,self.img_rows, self.img_cols))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100,)
        #noise_shape = (self.channels, self.img_rows, self.img_cols)
        
        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_shape=noise_shape))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8)) 
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(3, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()
        
        images = []
        classes = ["0","1",'2']

        for cat in classes:
            TRAIN_DIR = "mias_mini-train" + '/' + cat
            TEST_DIR = "mias_mini-test" +'/' + cat

            for imgName in os.listdir(TRAIN_DIR):
                if "png" in imgName:
                    images.append(cv2.imread(TRAIN_DIR + '/' + imgName))

        X_train = np.array(images[1:])  

  


        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Sample noise and generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, 100))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            # Plot the progress
            #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)
        
        print ("shape is", gen_imgs.shape)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        #fig.suptitle("DCGAN: Generated digits", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("dcgan/images/mias_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)

get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import matplotlib.pyplot as plt


img = cv2.imread('dcgan/images/mnist_0.png')

plt.imshow(img)
plt.show()

img.shape

get_ipython().run_cell_magic('writefile', 'dcgan.py', 'from keras.models import Sequential\nfrom keras.layers import Dense\nfrom keras.layers import Reshape\nfrom keras.layers.core import Activation\nfrom keras.layers.normalization import BatchNormalization\nfrom keras.layers.convolutional import UpSampling2D\nfrom keras.layers.convolutional import Conv2D, MaxPooling2D\nfrom keras.layers.core import Flatten\nfrom keras.optimizers import SGD\nfrom keras.datasets import mnist\nimport numpy as np\nfrom PIL import Image\nimport argparse\nimport math\n\n\ndef generator_model():\n    model = Sequential()\n    model.add(Dense(input_dim=100, output_dim=1024))\n    model.add(Activation(\'tanh\'))\n    model.add(Dense(128*7*7))\n    model.add(BatchNormalization())\n    model.add(Activation(\'tanh\'))\n    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))\n    model.add(UpSampling2D(size=(2, 2)))\n    model.add(Conv2D(64, (5, 5), padding=\'same\'))\n    model.add(Activation(\'tanh\'))\n    model.add(UpSampling2D(size=(2, 2)))\n    model.add(Conv2D(1, (5, 5), padding=\'same\'))\n    model.add(Activation(\'tanh\'))\n    return model\n\n\ndef discriminator_model():\n    model = Sequential()\n    model.add(\n            Conv2D(64, (5, 5),\n            padding=\'same\',\n            input_shape=(28, 28, 1))\n            )\n    model.add(Activation(\'tanh\'))\n    model.add(MaxPooling2D(pool_size=(2, 2)))\n    model.add(Conv2D(128, (5, 5)))\n    model.add(Activation(\'tanh\'))\n    model.add(MaxPooling2D(pool_size=(2, 2)))\n    model.add(Flatten())\n    model.add(Dense(1024))\n    model.add(Activation(\'tanh\'))\n    model.add(Dense(1))\n    model.add(Activation(\'sigmoid\'))\n    return model\n\n\ndef generator_containing_discriminator(g, d):\n    model = Sequential()\n    model.add(g)\n    d.trainable = False\n    model.add(d)\n    return model\n\n\ndef combine_images(generated_images):\n    num = generated_images.shape[0]\n    width = int(math.sqrt(num))\n    height = int(math.ceil(float(num)/width))\n    shape = generated_images.shape[1:3]\n    image = np.zeros((height*shape[0], width*shape[1]),\n                     dtype=generated_images.dtype)\n    for index, img in enumerate(generated_images):\n        i = int(index/width)\n        j = index % width\n        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \\\n            img[:, :, 0]\n    return image\n\n\ndef train(BATCH_SIZE):\n    (X_train, y_train), (X_test, y_test) = mnist.load_data()\n    X_train = (X_train.astype(np.float32) - 127.5)/127.5\n    X_train = X_train[:, :, :, None]\n    X_test = X_test[:, :, :, None]\n    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])\n    d = discriminator_model()\n    g = generator_model()\n    d_on_g = generator_containing_discriminator(g, d)\n    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)\n    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)\n    g.compile(loss=\'binary_crossentropy\', optimizer="SGD")\n    d_on_g.compile(loss=\'binary_crossentropy\', optimizer=g_optim)\n    d.trainable = True\n    d.compile(loss=\'binary_crossentropy\', optimizer=d_optim)\n    for epoch in range(100):\n        print("Epoch is", epoch)\n        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))\n        for index in range(int(X_train.shape[0]/BATCH_SIZE)):\n            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))\n            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]\n            generated_images = g.predict(noise, verbose=0)\n            if index % 20 == 0:\n                image = combine_images(generated_images)\n                image = image*127.5+127.5\n                Image.fromarray(image.astype(np.uint8)).save(\n                    str(epoch)+"_"+str(index)+".png")\n            X = np.concatenate((image_batch, generated_images))\n            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE\n            d_loss = d.train_on_batch(X, y)\n            print("batch %d d_loss : %f" % (index, d_loss))\n            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))\n            d.trainable = False\n            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)\n            d.trainable = True\n            print("batch %d g_loss : %f" % (index, g_loss))\n            if index % 10 == 9:\n                g.save_weights(\'generator\', True)\n                d.save_weights(\'discriminator\', True)\n\n\ndef generate(BATCH_SIZE, nice=False):\n    g = generator_model()\n    g.compile(loss=\'binary_crossentropy\', optimizer="SGD")\n    g.load_weights(\'generator\')\n    if nice:\n        d = discriminator_model()\n        d.compile(loss=\'binary_crossentropy\', optimizer="SGD")\n        d.load_weights(\'discriminator\')\n        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))\n        generated_images = g.predict(noise, verbose=1)\n        d_pret = d.predict(generated_images, verbose=1)\n        index = np.arange(0, BATCH_SIZE*20)\n        index.resize((BATCH_SIZE*20, 1))\n        pre_with_index = list(np.append(d_pret, index, axis=1))\n        pre_with_index.sort(key=lambda x: x[0], reverse=True)\n        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)\n        nice_images = nice_images[:, :, :, None]\n        for i in range(BATCH_SIZE):\n            idx = int(pre_with_index[i][1])\n            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]\n        image = combine_images(nice_images)\n    else:\n        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))\n        generated_images = g.predict(noise, verbose=1)\n        image = combine_images(generated_images)\n    image = image*127.5+127.5\n    Image.fromarray(image.astype(np.uint8)).save(\n        "dcgan/generated_image.png")\n\n\ndef get_args():\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--mode", type=str)\n    parser.add_argument("--batch_size", type=int, default=128)\n    parser.add_argument("--nice", dest="nice", action="store_true")\n    parser.set_defaults(nice=False)\n    args = parser.parse_args()\n    return args\n\nif __name__ == "__main__":\n    args = get_args()\n    if args.mode == "train":\n        train(BATCH_SIZE=args.batch_size)\n    elif args.mode == "generate":\n        generate(BATCH_SIZE=args.batch_size, nice=args.nice)')

get_ipython().system('python3.5 dcgan.py --mode generate --batch_size 128 --nice')

from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import numpy as np
import cv2

images = []
classes = ["1"]

for cat in classes:
    TRAIN_DIR = "mias_mini-train" + '/' + cat
    TEST_DIR = "mias_mini-test" +'/' + cat

    for imgName in os.listdir(TRAIN_DIR):
        if "png" in imgName:
            images.append(cv2.imread(TRAIN_DIR + '/' + imgName))

X_train = np.array(images[1:])  

X_train.shape

generator = Sequential([
        Dense(64*64*64, input_dim=100),
        LeakyReLU(0.2),
        BatchNormalization(),
        Reshape((64,64,64)),
        UpSampling2D(),
        Conv2D(32, (5, 5), padding='same'),
        LeakyReLU(0.2),
        BatchNormalization(),
        UpSampling2D(),
        Conv2D(3, (5, 5), padding='same'),
        Activation('tanh'),
    ])

generator.summary()

discriminator = Sequential([
        Conv2D(32, (5, 5), strides=(2,2), input_shape=(256,256,3), padding='same'),
        LeakyReLU(0.2),
        Dropout(0.3),
        Conv2D(64, (5, 5), strides=(2,2), padding='same'),
        LeakyReLU(0.2),
        Dropout(0.3),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

discriminator.summary()

generator.compile(loss='binary_crossentropy', optimizer=Adam())
discriminator.compile(loss='binary_crossentropy', optimizer=Adam())

discriminator.trainable = False
ganInput = Input(shape=(100,))
# getting the output of the generator
# and then feeding it to the discriminator
# new model = D(G(input))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

gan.summary()

def train(epoch=10, batch_size=128):
    batch_count = X_train.shape[0] // batch_size
    
    for i in range(epoch):
        for j in tqdm(range(batch_count)):
            # Input for the generator
            noise_input = np.random.rand(batch_size, 100)
            
            # getting random images from X_train of size=batch_size 
            # these are the real images that will be fed to the discriminator
            image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
            
            # these are the predicted images from the generator
            predictions = generator.predict(noise_input, batch_size=batch_size)
            
            # the discriminator takes in the real images and the generated images
            X = np.concatenate([predictions, image_batch])
            
            # labels for the discriminator
            y_discriminator = [0]*batch_size + [1]*batch_size
            
            # Let's train the discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_discriminator)
            
            # Let's train the generator
            noise_input = np.random.rand(batch_size, 100)
            y_generator = [1]*batch_size
            discriminator.trainable = False
            gan.train_on_batch(noise_input, y_generator)

train(30, 128)

generator.save_weights('gen_30_scaled_images.h5')
discriminator.save_weights('dis_30_scaled_images.h5')

train(20, 128)

generator.save_weights('gen_50_scaled_images.h5')
discriminator.save_weights('dis_50_scaled_images.h5')

generator.load_weights('gen_50_scaled_images.h5')
discriminator.load_weights('dis_50_scaled_images.h5')

def plot_output():
    try_input = np.random.rand(100, 100)
    preds = generator.predict(try_input)

    plt.figure(figsize=(10,10))
    for i in range(preds.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(preds[i, :, :, 0], cmap='gray')
        plt.axis('off')
    
    # tight_layout minimizes the overlap between 2 sub-plots
    plt.tight_layout()


plot_output()

generator = Sequential([
        Dense(128*8*8, input_dim=100),
        LeakyReLU(0.2),
        BatchNormalization(),
        Reshape((8,8,128)),
        UpSampling2D(),
        Conv2D(64, (5, 5), padding='same'),
        LeakyReLU(0.2),
        BatchNormalization(),
        UpSampling2D(),
        Conv2D(32, (5, 5), padding='same'),
        LeakyReLU(0.2),
        BatchNormalization(),
        UpSampling2D(),
        Conv2D(16, (5, 5), padding='same'),
        LeakyReLU(0.2),
        BatchNormalization(),
        UpSampling2D(),
        Conv2D(8, (5, 5), padding='same'),
        LeakyReLU(0.2),
        BatchNormalization(),
        UpSampling2D(),
        Conv2D(1, (5, 5), padding='same'),
        Activation('tanh'),
    ])

generator.summary()

discriminator = Sequential([
        Conv2D(32, (5, 5), strides=(2,2), input_shape=(256,256,1), padding='same'),
        LeakyReLU(0.2),
        Dropout(0.3),
        Conv2D(64, (5, 5), strides=(2,2), padding='same'),
        LeakyReLU(0.2),
        Dropout(0.3),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

discriminator.summary()

generator.compile(loss='binary_crossentropy', optimizer=Adam())
discriminator.compile(loss='binary_crossentropy', optimizer=Adam())

discriminator.trainable = False
ganInput = Input(shape=(100,))
# getting the output of the generator
# and then feeding it to the discriminator
# new model = D(G(input))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

gan.summary()

X_train = X_train.reshape(86,256,256,1)

def train(epoch=10, batch_size=128):
    batch_count = X_train.shape[0] // batch_size
    
    for i in range(epoch):
        for j in tqdm(range(batch_count)):
            # Input for the generator
            noise_input = np.random.rand(batch_size, 100)
            
            # getting random images from X_train of size=batch_size 
            # these are the real images that will be fed to the discriminator
            image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
            
            # these are the predicted images from the generator
            predictions = generator.predict(noise_input, batch_size=batch_size)
            
            # the discriminator takes in the real images and the generated images
            X = np.concatenate([predictions, image_batch])
            
            # labels for the discriminator
            y_discriminator = [0]*batch_size + [1]*batch_size
            
            # Let's train the discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_discriminator)
            
            # Let's train the generator
            noise_input = np.random.rand(batch_size, 100)
            y_generator = [1]*batch_size
            discriminator.trainable = False
            gan.train_on_batch(noise_input, y_generator)

train(30, 20)

generator.save_weights('gen1_30_scaled_images.h5')
discriminator.save_weights('dis1_30_scaled_images.h5')

train(30, 20)

generator.save_weights('gen1_50_scaled_images.h5')
discriminator.save_weights('dis1_50_scaled_images.h5')

generator.load_weights('gen1_50_scaled_images.h5')
discriminator.load_weights('dis1_50_scaled_images.h5')

def plot_output():
    try_input = np.random.rand(100, 100)
    preds = generator.predict(try_input)

    plt.figure(figsize=(40,40))
    for i in range(preds.shape[0]):
        plt.subplot(20, 5, i+1)
        plt.imshow(preds[i, :, :,:].reshape(256,256), cmap ='gray')
        plt.axis('off')
    
    # tight_layout minimizes the overlap between 2 sub-plots
    plt.tight_layout()


plot_output()


plot_output()

X_train.shape

def plot_input():
    try_input = np.random.rand(100, 100)
    preds = generator.predict(try_input)

    plt.figure(figsize=(40,40))
    for i in range(X_train.shape[0]):
        plt.subplot(20, 5, i+1)
        plt.imshow(X_train[i, :, :,:].reshape(256,256), cmap ='gray')
        plt.axis('off')
    
    # tight_layout minimizes the overlap between 2 sub-plots
    plt.tight_layout()


plot_input()

try_input = np.random.rand(100, 100)
preds = generator.predict(try_input)

preds[0].shape

np.min(preds[0,:,:,1])

img = preds[0].reshape(256,256)

plt.imshow(img, cmap ='gray')

