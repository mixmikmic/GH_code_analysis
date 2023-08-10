from nn import get_layer_sizes
from nn import draw_neural_net
from nn import nn_to_formula_components
from nn import formula_components_to_dnf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier # multi-layer perceptron (MLP)

# Training set
X_img = [
     [[0, 0],     [0, 0]],
     [[255, 255], [255, 255]],
     [[0, 0],     [0, 255]],
     [[0, 0],     [255, 0]],
     [[0, 255],   [0, 0]],
     [[255, 0],   [0, 0]],
     [[255, 255], [0, 0]],
     [[255, 0],   [255, 0]],
     [[255, 0],   [0, 255]],
     [[0, 255],   [255,0]],
     [[0, 255],   [0,255]],
     [[0, 0],     [255,255]],
     [[0, 255],   [255,255]],
     [[255, 0],   [255,255]],
     [[255, 255], [0,255]],
     [[255, 255], [255,0]],  
    ]
y = [1 if img[1][0] <= 50 and img[1][1] <= 50 else 0 for img in X_img]
print(y)

# Gray scale from black (0) to white (255)
norm = norm=matplotlib.colors.NoNorm(vmin=0, vmax=255, clip=True)

fig = plt.figure(figsize=(10,10))
for idx,img in enumerate(X_img):
    if (idx > 15): break
    arr = np.array(img)
    ax = plt.subplot(5,4,idx+1)
    plt.imshow(arr, cmap='gray', norm=norm)
    if y[idx]:
        ax.set_title("Y=1", color='red')
        ax.tick_params(axis='x', colors='red')
        ax.tick_params(axis='y', colors='red')     
fig.tight_layout()
plt.show()

# Defining the neural network
hidden_layer_sizes = (2,)
clf = MLPClassifier(
            solver='lbfgs',    # solver used when training
            alpha=1e-5,        # L2 penalty (regularization term) parameter
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=1,    # If int, it's the seed of the random number gen.
            activation="relu") # activation function
# Turing images into vector
X = [[float(item) for item in np.concatenate(img)] for img in X_img]
# Train the neural network
clf.fit(X, y)
print(clf)
print("Layers", clf.n_layers_)
print("Coefs", clf.coefs_)
print("Intercepts", clf.intercepts_)
print("n inter", clf.n_iter_)
print("loss", clf.loss_)

# Draw the neural network
fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
ax.axis('off')
layer_sizes = get_layer_sizes(clf.coefs_, clf.intercepts_)
print(layer_sizes)
draw_neural_net(ax, .1, .9, .1, .9,
    clf.coefs_,
    clf.intercepts_,
    clf.n_iter_,
    clf.loss_)

formula_components = nn_to_formula_components(clf.coefs_, clf.intercepts_)
print("Linear part of each neuron, that is, before activation function:")
for (var,val) in formula_components:
    print("%s = %s" % (var,val))

dnf_formula = formula_components_to_dnf(formula_components)
print("DNF formula for neural network:")
for clause in dnf_formula:
    for term in clause:
        print(term)
    print()

ori_img = [[0,0],[0,0]]
imgs = [ori_img]
predictions = []
for img in imgs:
    vec = [float(item) for item in np.concatenate(img)]
    pred = clf.predict([vec])
    print("Vector ", vec)
    print("Pred.  ", pred)
    print()
    predictions.append(pred)

fig = plt.figure(figsize=(10,10))
for idx,img in enumerate(imgs):
    if (idx > 15): break
    arr = np.array(img)
    ax = plt.subplot(5,4,idx+1)
    ax.set_title(predictions[idx])
    plt.imshow(arr, cmap='gray', norm=norm)
fig.tight_layout()
plt.show()

in_img_str = ["X%d==%f + D%d" % (idx,val,idx) for idx,val in enumerate(np.concatenate(ori_img))]
print(in_img_str)

from scipy.optimize import linprog

c = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
A_ub = [
    [-0.1655, -0.9968, -0.7044, -0.6256, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -0.2279, 1.7546, 0, 0, 0, 0, 0]
]
b_ub = [0.2065, 1.3866]
A_eq = [
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0.4507, 0.3895, 0.8129, 1.2027, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1]
]
b_eq = [0, 6.4341, 0, 0, 0, 0]
x_bounds = (0, 255)
y_bounds = (None, None)
d_bounds = (0, 3)
bounds = [
    x_bounds, x_bounds, x_bounds, x_bounds, # X0, X1, X2, X3
    (None,None), # Y0_0, There is already an equation for Y0_0==0
    (0,None),    # Y0_1, Y0_1 > 0 from system of linear equations
    (None,0),    # Y1_0, Y1_0 < 0 for misclassification
    d_bounds, d_bounds, d_bounds, d_bounds, # D0, D1, D2, D3
]
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, options={"disp": True})
print(res)

for idx,vec in enumerate(X):
    pred = clf.predict([vec])
    if pred == 1:
        print("Vector ", vec)
        print("Label  ", y[idx])
        print("Pred.  ", pred)
        print()
test_set = [
    [[0,0],[0,0]],
    [[0,0],[2,2]],
    [[0,0],[3,3]],    
]
predictions = []
for img in test_set:
    vec = [float(item) for item in np.concatenate(img)]
    pred = clf.predict([vec])
    predictions.append(pred)

fig = plt.figure(figsize=(10,10))
for idx,img in enumerate(test_set):
    if (idx > 15):
        break
    arr = np.array(img)
    ax = plt.subplot(5,4,idx+1)
    ax.set_title("%s\n%s" % (img, predictions[idx]))
    plt.imshow(arr, cmap='gray', norm=norm)
fig.tight_layout()
plt.show()

