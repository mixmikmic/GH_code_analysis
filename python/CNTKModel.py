# All the necessary imports
import cntk
from tqdm import tqdm
from captcha.image import ImageCaptcha
import string
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random

characters = string.digits + string.ascii_uppercase
n_len = 4
width, height, n_class, batch_size = 170, 80, len(characters), 32

def data_generator(batch_size=64):
    X = np.zeros((batch_size, 3, height, width), dtype=np.float32)
    y = [np.zeros((batch_size, n_class), dtype=np.float32) for i in range(n_len)]
    
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(n_len)])
            generator = ImageCaptcha(width=width, height=height)
            image = generator.generate_image(random_str)
            # Reshape it in a way that CNTK and our model understands
            X[i] = np.asarray(image).reshape((3, height, width))
            # One hot encoding the output labels.
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

X, y = next(data_generator(1))
plt.imshow(X[0].reshape((height, width, 3)))
plt.title(decode(y))

def create_model(features):
    h = features
    with cntk.layers.default_options(init=cntk.glorot_uniform(), activation=cntk.relu):
        for i in range(4):
            h = cntk.layers.Convolution2D((3, 3), num_filters=32*2**1, name='Conv{}'.format(i))(h)
            h = cntk.layers.Convolution2D((3, 3), num_filters=32*2**1, name='Conv{}'.format(i))(h)

            h = cntk.layers.MaxPooling(filter_shape=(2, 2), name='MaxPool{}'.format(i))(h)
            
        h = cntk.layers.Dropout(dropout_rate=0.25)(h)
    with cntk.layers.default_options(activation=None):
        h1 = cntk.layers.Dense(n_class, name='Dense1')(h)
        
        h2 = cntk.layers.Dense(n_class, name='Dense2')(h)
        
        h3 = cntk.layers.Dense(n_class, name='Dense3')(h)
        
        h4 = cntk.layers.Dense(n_class, name='Dense4')(h)
        
        final_model = cntk.combine(h1, h2, h3, h4)
        return final_model

input_placeholder = cntk.input_variable(shape=(3, height, width))
output_placeholder0 = cntk.input_variable(shape=n_class)
output_placeholder1 = cntk.input_variable(shape=n_class)
output_placeholder2 = cntk.input_variable(shape=n_class)
output_placeholder3 = cntk.input_variable(shape=n_class)

z = create_model(input_placeholder/255.0)
cntk.logging.plot(z, 'cntk_model.png')

loss0 = cntk.cross_entropy_with_softmax(z.outputs[0], output_placeholder0)
loss1 = cntk.cross_entropy_with_softmax(z.outputs[1], output_placeholder1)
loss2 = cntk.cross_entropy_with_softmax(z.outputs[2], output_placeholder2)
loss3 = cntk.cross_entropy_with_softmax(z.outputs[3], output_placeholder3)

loss = loss0 + loss1 + loss2 + loss3

label_error1 = cntk.classification_error(z.outputs[0], output_placeholder0)
label_error2 = cntk.classification_error(z.outputs[1], output_placeholder1)
label_error3 = cntk.classification_error(z.outputs[2], output_placeholder2)
label_error4 = cntk.classification_error(z.outputs[3], output_placeholder3)

label_error = label_error1 + label_error2 + label_error3 + label_error4

learning_rate = 1.0

learner = cntk.adadelta(z.parameters, lr=learning_rate, rho=0.95)

num_epox = 100
num_batches_per_epoch = 800

trainer = cntk.Trainer(z, (loss, label_error), [learner])

def test_with_image():
    test_x, test_y = next(data_generator(1))
    evaled = z.eval({input_placeholder:test_x})
    temp = []
    for key, value in evaled.items():
        temp.append(value)
    out = cntk.softmax(temp).eval()
    plt.title('real: %s\npred:%s'%(decode(test_y), decode(out)))
    plt.imshow(test_x[0].reshape((height, width, 3)), cmap='gray')
    plt.axis('off')
    plt.show()

for epoch_number in range(num_epox):
    for batch in tqdm(
            range(num_batches_per_epoch),
            ncols=90,
            smoothing=1,
            desc='Epoch {}/{}'.format((epoch_number + 1), num_epox)):
        x, y = next(data_generator(batch_size))
        
        minibatch_losses = []
        trainer.train_minibatch(
            {
                input_placeholder: x,
                output_placeholder0: y[0],
                output_placeholder1: y[1],
                output_placeholder2: y[2],
                output_placeholder3: y[3]
            }
        )
        minibatch_losses.append(trainer.previous_minibatch_loss_average)
    print('Average training loss after {0} epoch out of {1}: {2}'.format((epoch_number + 1), num_epox, np.mean(minibatch_losses)))

test_with_image()

z.save('model_weights.dnn')



