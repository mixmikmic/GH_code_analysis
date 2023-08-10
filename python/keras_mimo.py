from ds_utils.imports import *

main_input = keras.layers.Input(shape=(1,))

embedding_layer = keras.layers.Embedding(input_dim=10, output_dim=1)

lstm_layer = keras.layers.LSTM(units=5)

aux_input = keras.layers.Input(shape=(1,))

aux_output = keras.layers.Dense(units=1)

merge_layer = keras.layers.Merge(layers=[aux_input, lstm_layer], mode='sum')

model = keras.models.Model()



