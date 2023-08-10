import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
get_ipython().magic('matplotlib inline')

# Load data
df = pd.read_csv('./data/versicolor/train.csv')
X = df[['petal length (cm)', 'petal width (cm)']].values
y = df['versicolor'].values

def plot_keras_model( model=None ):
    "Plot the Keras model, along with data"
    plt.clf()
    # Calculate the probability on a mesh
    if model is not None:
        petal_width_mesh, petal_length_mesh =             np.meshgrid( np.linspace(0,3,100), np.linspace(0,8,100) )
        petal_width_mesh = petal_width_mesh.flatten()
        petal_length_mesh = petal_length_mesh.flatten()
        p = model.predict( np.stack( (petal_length_mesh, petal_width_mesh), axis=1 ) )
        p = p.reshape((100,100))
        # Plot the probability on the mesh
        plt.imshow( p.T, extent=[0,8,0,3], origin='lower', 
               vmin=0, vmax=1, cmap='RdBu', aspect='auto', alpha=0.7 )
    # Plot the data points
    plt.scatter( df['petal length (cm)'], df['petal width (cm)'], c=df['versicolor'], cmap='RdBu')
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    cb = plt.colorbar()
    cb.set_label('versicolor')
plot_keras_model()

# Build the model
single_layer_model = Sequential()
single_layer_model.add( Dense( output_dim=1, input_dim=2 ) )
single_layer_model.add( Activation( 'sigmoid' ) )

# Prepare the model for training
single_layer_model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

# Train the network
single_layer_model.fit( X, y, batch_size=16, nb_epoch=1000, verbose=0 )

plot_keras_model( model=single_layer_model )

# Build the model: pick 8 units in the intermediate layer
two_layer_model = Sequential()
two_layer_model.add( Dense( output_dim=8, input_dim=2 ) )
two_layer_model.add( Activation( 'sigmoid' ) )
two_layer_model.add( Dense( output_dim=1, input_dim=8 ) )
two_layer_model.add( Activation( 'sigmoid' ) )

# Compile the model
two_layer_model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

# Train it
two_layer_model.fit( X, y, batch_size=16, nb_epoch=1000, verbose=0 )

plot_keras_model( model=two_layer_model )

