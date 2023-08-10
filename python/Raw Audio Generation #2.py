from helpers.extractor import *
from helpers.sound_tools import *
from helpers.neural_network import *

sounds = [
              "inputs/sounds/flower.wav", 
              "inputs/sounds/3129775.wav", 
              "inputs/sounds/sound1.wav",
              "inputs/sounds/139368801.wav",
              "inputs/sounds/139364551.wav",
              "inputs/sounds/90781157.wav",
         ]

builder = []
for sound in sounds:
    builder.append(SoundLoader(sound=sound,
                         fixed_size=32*32, 
                         sample_size=0.01, 
                         #insert_global_input_state = 4,
                         one_hot = 256,
                         uLaw = 256,
                         amplitude = 0.5,
                         samplerate = 4410,
                         n_steps = 32,
                         ))

loader = SoundCombiner(builder)

print(loader.fixed_size)

network = ConvNet(loader, 
                  training_iters=2000000, 
                  display_step=10, 
                  save_step = 30000,
                  learning_rate = 0.001, 
                  decay_step = 10000000,
                  decay_rate = 0.95,
                  batch_size = loader.batch_size,
                  n_steps = loader.n_steps,
                 )

print(network.n_input)
print(network.n_classes)

x=tf.placeholder(tf.float32, [None, 32, 32], "X")
#x=tf.placeholder("float", [None, 8*8])

gru = GRUOperation(cells=[1024]*3, n_classes=network.n_classes)

layers = []
layers.append(gru)

network.Run(layers, 
            x=x, 
            #restore_path="graphs/MultiAudioGeneration", 
            save_path="graphs/MultiAudioGeneration",
            state = gru.state,
           )

network.Plot()

from helpers.extractor import *
from helpers.sound_tools import *
from helpers.neural_network import *

loader = SoundLoader(sound="inputs/sounds/139364551.wav",
                     fixed_size=32*32, 
                     sample_size=0.01, 
                     #insert_global_input_state = 4,
                     one_hot = 256,
                     uLaw = 256,
                     samplerate=4410,
                     random=True,
                     amplitude=0.5,
                     )

network = ConvNet(loader, n_steps=32)

x=tf.placeholder("float", [None, 32, 32])

gru = GRUOperation(cells=[1024]*3, n_classes=network.n_classes)

#--------------------------------------------------------------------------------------------
#Layers
layers = []
layers.append(gru)
#End layers
#--------------------------------------------------------------------------------------------

batch = loader.getNextTimeBatch(1, n_steps=network.n_steps)

print(loader.converter.last_sample_position)

prediction = network.Generate(batch[0][0], 
                              "graphs/MultiAudioGeneration", 
                              x = x,
                              layers = layers, 
                              iterations=4410*1, 
                              display_step = 100,
                              epsilon = 0, 
                              state = gru.state
                             )

print(loader.converter.last_sample_position)

from helpers.extractor import *
from helpers.sound_tools import *
from helpers.neural_network import *
converter = SoundConverter("")
data = converter.TensorToSound(prediction, "outputs/generated2.wav", multiplier=1, offset=0, samplerate=loader.samplerate)

import IPython
IPython.display.Audio("outputs/generated2.wav")

import matplotlib.pyplot as plt
plt.plot(prediction[0:loader.fixed_size])
plt.title("Activation")
plt.show()
plt.plot(prediction[loader.fixed_size:])
plt.title("Generated")
plt.show()



