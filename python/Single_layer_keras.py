import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Keras imports
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# Build the model with keras
model = Sequential()
model.add( Dense( output_dim=1, input_dim=2 ) )
model.add( Activation( 'sigmoid' ) )

# Print the summary
model.summary()

# Load data
df = pd.read_csv('./data/setosa/train.csv')
X = df[['petal length (cm)', 'petal width (cm)']].values
y = df['setosa'].values

def plot_keras_model():
    "Plot the results of the model, along with the data points"
    # Calculate the probability on a mesh
    petal_width_mesh, petal_length_mesh =         np.meshgrid( np.linspace(0,3,100), np.linspace(0,8,100) )
    petal_width_mesh = petal_width_mesh.flatten()
    petal_length_mesh = petal_length_mesh.flatten()
    p = model.predict( np.stack( (petal_length_mesh, petal_width_mesh), axis=1 ) )
    p = p.reshape((100,100))
    # Plot the probability on the mesh
    plt.clf()
    plt.imshow( p.T, extent=[0,8,0,3], origin='lower', 
               vmin=0, vmax=1, cmap='RdBu', aspect='auto', alpha=0.7 )
    # Plot the data points
    plt.scatter( df['petal length (cm)'], df['petal width (cm)'], c=df['setosa'], cmap='RdBu')
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    cb = plt.colorbar()
    cb.set_label('setosa')
plot_keras_model()

# Prepare the model for training
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

# Train the network
model.fit( X, y, batch_size=16, nb_epoch=20, verbose=1 )

plot_keras_model()

df_test = pd.read_csv('./data/setosa/test.csv')
df_test.head(10)

model.predict( np.array([[4.2, 1.5]]) )

df_test['probability_setosa_predicted'] = model.predict( df_test[['petal length (cm)', 'petal width (cm)']].values )

df_test

