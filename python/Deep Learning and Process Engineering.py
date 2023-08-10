import numpy as np
from keras.models import Model, Sequential,load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
np.set_printoptions(precision=6,suppress=True)
get_ipython().magic('matplotlib inline')

#Global parameters
epocas = 3000 # Global variable controlling the number of epochs for all ML models
n = 150 # Number of Patterns to be parametrically generated (multiple of 3)
sd1=.25 # Standard Deviation for the generation ofprocess variables
sd2 = 0.015 # Standard Deviation for the intrumental noise
tr=.05 # Normal operation treshold

# Operating conditions (OC)
def cond(p1,p2,flag): 
    if flag==1: # Normal OC (acceptable p1 and p2)
        return(abs(p1)<tr and abs(p2)<tr)
    elif flag==2: #Type 1 error (acceptable p2, unacceptable p1)
        return(abs(p1)>tr and abs(p2)<tr)
    elif flag==3: #Type 2 error (acceptable p1, unacceptable p2)
        return(abs(p1)<tr and abs(p2)>tr)

patterns = np.zeros([n,5]) # Parametric matrix to generate database
count = 0
i=[0,int(n/3),int(2*n/3),int(n)] # Equally sized OC indexes. Normal: [0:n/3], T1: [n/3:2n/3], T2: [2n:3,n]
np.random.seed(1234)
for pos,idx in enumerate(i):
    while count < idx:
        # Random (p1,p2)
        p1 = np.random.normal(0,sd1)
        p2 = np.random.normal(0,sd1)
        # Verify process state
        if cond(p1,p2,pos):
            # Record operation condition
            patterns[count,0] = p1
            patterns[count,1] = p2
            patterns[count,pos+1] = 1
            count += 1
# Plot OCs
plt.scatter(patterns[i[0]:i[1],0],patterns[i[0]:i[1],1],c='r',s=2)
plt.scatter(patterns[i[1]:i[2],0],patterns[i[1]:i[2],1],c='b',s=2)
plt.scatter(patterns[i[2]:i[3],0],patterns[i[2]:i[3],1],c='g',s=2)
plt.show()

plt.scatter(np.absolute(patterns[i[0]:i[1],0]),np.absolute(patterns[i[0]:i[1],1]),c='r',s=2)
plt.scatter(np.absolute(patterns[i[1]:i[2],0]),np.absolute(patterns[i[1]:i[2],1]),c='b',s=2)
plt.scatter(np.absolute(patterns[i[2]:i[3],0]),np.absolute(patterns[i[2]:i[3],1]),c='g',s=2)
plt.title("Generational data in the (S,tan) space")
plt.show()

x = np.zeros([n,5])
for idi in range(len(patterns)):
    x[idi,0] = patterns[idi,0] + patterns[idi,1]
    x[idi,1] = patterns[idi,0] - patterns[idi,1]
    x[idi,2:] = patterns[idi,2:]
plt.scatter(x[i[0]:i[1],0],x[i[0]:i[1],1],c='r',s=2)
plt.scatter(x[i[1]:i[2],0],x[i[1]:i[2],1],c='b',s=2)
plt.scatter(x[i[2]:i[3],0],x[i[2]:i[3],1],c='g',s=2)
plt.show()

data = x[:,0:2]
target = x[:,2:]

get_ipython().run_cell_magic('time', '', 'np.random.seed(2379)\nmodel = Sequential() # Fully connected ANN\n# Adding the layers to design the 5-3-2-2-3 architecture\n# Hyperblic tangent activation functions in all layers\nmodel.add(Dense(units=5, activation=\'tanh\', input_dim=2,name="l4"))\nmodel.add(Dense(units=3, activation=\'tanh\',name="l3"))\nmodel.add(Dense(units=2, activation=\'tanh\',name="l2"))\nmodel.add(Dense(units=2, activation=\'tanh\',name="l1"))\nmodel.add(Dense(units=3, activation=\'softmax\'))\n# Defining ANN optimizer, error and pperformance\nmodel.compile(loss=\'categorical_crossentropy\',\n              optimizer=\'sgd\',\n              metrics=[\'accuracy\'])\n# Early stopping. Saves the best performance model\ncallbacks = ModelCheckpoint(\'bestfit.hdf5\',save_best_only=True)\n# Saving performance curves and training the model\nhistory = model.fit(data, target, epochs=epocas, batch_size=1,verbose=0,callbacks=[callbacks],validation_data=(data,target))\n# PLotting training performance\nfig = plt.figure(figsize=(10,5))\nax1 = fig.add_subplot(121)\nax2 = fig.add_subplot(122)\nax1.plot(history.history[\'loss\'])\nax1.plot(history.history[\'val_loss\'])\nax2.plot(history.history[\'acc\'])\nax2.plot(history.history[\'val_acc\'])\nplt.show()\n# Writing performance indicators and confusion matrix on the screen\nmodel = load_model(\'bestfit.hdf5\')\nevaluation = model.evaluate(data,target,verbose=0)\nprint("Error: ",evaluation[0])\nprint("Accuracy: ",evaluation[1])\ny_pred = model.predict(data)\nCM = confusion_matrix(np.argmax(target,axis=1),np.argmax(y_pred,axis=1))\nprint("Confusion C_(i,j) -> i: predicted, j: actual. ")\nprint(CM)')

# Retrieving the outputs in the last layer space (LLS)
inter = Model(inputs=model.input, outputs=model.get_layer('l1').output)
interout = inter.predict(data)
# Plotting  the ANN outputs in the LLS
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(interout[i[0]:i[1],1],interout[i[0]:i[1],0],'ro',
         interout[i[1]:i[2],1],interout[i[1]:i[2],0],'bo',
         interout[i[2]:i[3],1],interout[i[2]:i[3],0],'go')
plt.show()

# Same comments as before apply
np.random.seed(2379)
model = Sequential()
model.add(Dense(units=9, activation='tanh', input_dim=2,name="l2"))
model.add(Dense(units=2, activation='tanh',name="l1"))
model.add(Dense(units=3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
callbacks = ModelCheckpoint('bestfit.hdf5',save_best_only=True)
history = model.fit(data, target, epochs=epocas, batch_size=1,verbose=0,callbacks=[callbacks],validation_data=(data,target))
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
ax2.plot(history.history['acc'])
ax2.plot(history.history['val_acc'])
plt.show()
model = load_model('bestfit.hdf5')
evaluation = model.evaluate(data,target,verbose=0)
print("Error: ",evaluation[0])
print("Accuracy: ",evaluation[1])
y_pred = model.predict(data)
CM = confusion_matrix(np.argmax(target,axis=1),np.argmax(y_pred,axis=1))
print("Confusion C_(i,j) -> i: predicted, j: actual. ")
print(CM)

inter = Model(inputs=model.input, outputs=model.get_layer('l1').output)
interout = inter.predict(data)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(interout[i[0]:i[1],1],interout[i[0]:i[1],0],'ro',
         interout[i[1]:i[2],1],interout[i[1]:i[2],0],'bo',
         interout[i[2]:i[3],1],interout[i[2]:i[3],0],'go')
plt.show()

count = 0
np.random.seed(1234)
for pos,idx in enumerate(i):
    while count < idx:
        p1 = np.random.normal(0,sd1)
        p2 = np.random.normal(0,sd1)
        if cond(p1,p2,pos):
            patterns[count,0] = p1 + np.random.normal(0,sd2) # sweetness plus noise
            patterns[count,1] = p2 + np.random.normal(0,sd2) # tannins plus noise
            patterns[count,pos+1] = 1
            count += 1
plt.scatter(patterns[i[0]:i[1],0],patterns[i[0]:i[1],1],c='r',s=2)
plt.scatter(patterns[i[1]:i[2],0],patterns[i[1]:i[2],1],c='b',s=2)
plt.scatter(patterns[i[2]:i[3],0],patterns[i[2]:i[3],1],c='g',s=2)
plt.show()

plt.scatter(np.absolute(patterns[i[0]:i[1],0]),np.absolute(patterns[i[0]:i[1],1]),c='r',s=2)
plt.scatter(np.absolute(patterns[i[1]:i[2],0]),np.absolute(patterns[i[1]:i[2],1]),c='b',s=2)
plt.scatter(np.absolute(patterns[i[2]:i[3],0]),np.absolute(patterns[i[2]:i[3],1]),c='g',s=2)
plt.title("Generational data in the (S,tan) space + noise")
plt.show()

x = np.zeros([n,5])
for idi in range(len(patterns)):
    x[idi,0] = patterns[idi,0] + patterns[idi,1]
    x[idi,1] = patterns[idi,0] - patterns[idi,1]
    x[idi,2:] = patterns[idi,2:]
data = x[:,0:2]
target = x[:,2:] 
# Everyday I'm shuffling! Pew, pew, pew, pew, pew-pada
xshuf = x[np.random.permutation(n)]
datashuf = xshuf[:,0:2]
targetshuf = xshuf[:,2:]
plt.scatter(x[i[0]:i[1],0],x[i[0]:i[1],1],c='r',s=2)
plt.scatter(x[i[1]:i[2],0],x[i[1]:i[2],1],c='b',s=2)
plt.scatter(x[i[2]:i[3],0],x[i[2]:i[3],1],c='g',s=2)
plt.show()

get_ipython().run_cell_magic('time', '', 'np.random.seed(2379)\nmodel = Sequential()\nmodel.add(Dense(units=5, activation=\'tanh\', input_dim=2,name="l4"))\nmodel.add(Dense(units=3, activation=\'tanh\',name="l3"))\nmodel.add(Dense(units=2, activation=\'tanh\',name="l2"))\nmodel.add(Dense(units=2, activation=\'tanh\',name="l1"))\nmodel.add(Dense(units=3, activation=\'softmax\'))\nmodel.compile(loss=\'categorical_crossentropy\',\n              optimizer=\'sgd\',\n              metrics=[\'accuracy\'])\ncallbacks = ModelCheckpoint(\'bestfit.hdf5\',save_best_only=True)\n# Training on the suffled data, 80-20 train-test split.\nhistory = model.fit(datashuf, targetshuf, epochs=int(epocas), batch_size=1,verbose=0,callbacks=[callbacks],validation_split=0.2)\nfig = plt.figure(figsize=(10,5))\nax1 = fig.add_subplot(121)\nax2 = fig.add_subplot(122)\nax1.plot(history.history[\'loss\'])\nax1.plot(history.history[\'val_loss\'])\nax2.plot(history.history[\'acc\'])\nax2.plot(history.history[\'val_acc\'])\nplt.show()\nmodel = load_model(\'bestfit.hdf5\')\nevaluation = model.evaluate(data,target,verbose=0)\nprint("Error: ",evaluation[0])\nprint("Accuracy: ",evaluation[1])\ny_pred = model.predict(data)\nCM = confusion_matrix(np.argmax(target,axis=1),np.argmax(y_pred,axis=1))\nprint("Confusion C_(i,j) -> i: predicted, j: actual. ")\nprint(CM)')

inter = Model(inputs=model.input, outputs=model.get_layer('l1').output)
interout = inter.predict(data)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(interout[i[0]:i[1],1],interout[i[0]:i[1],0],'ro',
         interout[i[1]:i[2],1],interout[i[1]:i[2],0],'bo',
         interout[i[2]:i[3],1],interout[i[2]:i[3],0],'go')
#ax1.set_title("l1")
plt.show()

