from helpers.extractor import *
from helpers.deezer_tools import *
from helpers.neural_network import *

loader = DeezerLoader(fixed_size=128*128+10, 
                      sample_size=1/32, 
                      local=True, 
                      limit_track=15781392, 
                      limit_genres=[113, 152], 
                      other_genres_rate=0,
                      LABELS_COUNT = 10,
                      #rating_labels=True
                     )
loader.shuffle_rate = 1
#loader.picker.FixAllTracks()

print(len(loader.picker.tracks))
print("tracks with genre 113: "+str(len(loader.picker.tracksByGenre[113])))

network = ConvNet(loader, 
                  training_iters=10000, 
                  display_step=2, 
                  learning_rate = 0.001, 
                  batch_size=128, 
                  n_steps=128,)
print(network.n_input)
print(network.n_classes)

layers = []
layers.append(LSTMOperation(cells=[512, 128, 32], n_classes=network.n_classes))

x=tf.placeholder("float", [None, 128, 128])

network.Run(layers, x=x, save_path="graphs/DeezerGraph", input_as_label=True)

network.Plot()

from helpers.extractor import *
from helpers.deezer_tools import *
from helpers.neural_network import *

loader = DeezerLoader(fixed_size=32*32, 
                      #extract_length = 2,
                      sample_size=1/16, 
                      local=True, 
                      #limit_track=15781392, 
                      limit_genres=[113, 152], 
                      other_genres_rate=0, 
                      #rating_labels=True,
                      #label_is_input = True,
                      #encoding="OneHot",
                      LABELS_COUNT = 2,
                      #insert_global_input_state = True,
                      )

loader.shuffle_rate = 1
#loader.picker.FixAllTracks()

network = ConvNet(loader, 
                  n_steps=32,
                  training_iters=10000, 
                  display_step=1, 
                  learning_rate = 0.0001, 
                  batch_size=64)
print(network.n_input)
print(network.n_classes)

x=tf.placeholder("float", [None, 32, 32])

layers = []
layers.append(NNOperation("reshape", [-1, 32, 32, 1]))
layers.append(NNOperation("conv2d", [3, 3, 1, 32]))
layers.append(NNOperation("maxpool2d", 2)) #16
layers.append(NNOperation("conv2d", [3, 3, 32, 64]))
layers.append(NNOperation("maxpool2d", 2)) #8
layers.append(NNOperation("conv2d", [3, 3, 64, 128]))
layers.append(NNOperation("maxpool2d", 2)) #4
layers.append(NNOperation("conv2d", [3, 3, 128, 256]))
layers.append(NNOperation("maxpool2d", 2)) #2
layers.append(NNOperation("reshape", [-1, 2*256, 2]))
layers.append(LSTMOperation(cells=[1024], n_classes=network.n_classes))

network.Run(layers=layers, x=x, save_path="graphs/DeezerLSTMGraph")

network.Plot()



