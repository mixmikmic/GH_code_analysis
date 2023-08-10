import matplotlib
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle code display on/off."></form>''')

# from http://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer

def generate_data(c1_mu, c2_mu, n, sigma=1.0):
    c1 = np.random.randn(n,2)*sigma+np.array(c1_mu)
    c2 = np.random.randn(n,2)*sigma+np.array(c2_mu)
    print "Generated data. Green points are from class 1, blue from class 2"
    plt.scatter(c1[:,0], c1[:,1], color='g')
    plt.scatter(c2[:,0], c2[:,1], color='b')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    return c1, c2

# Generate low-low high-high data.
n = 5
low = -5
high = 5
c1_mu = np.array([low, low])
c2_mu = np.array([high, high])
c1, c2 = generate_data(c1_mu, c2_mu, n)

def get_train_pt(c1, c2, i):
    if np.random.rand() < 0.5:
        return c1[i, :], -1
    else:
        return c2[i, :], 1

def pts_on_the_line(W, b):
    # Solve w[0]*x + w[1]*y + b for the value of y.
    if (W[1] == W[0] == 0): return 0,0,0,0
    if (W[1] == 0): return -b/W[0], -10, -b/W[0], 10
    x1 = -10
    y1 = -W[0]/W[1] * x1 - b/W[1]
    x2 = 10
    y2 = -W[0]/W[1] * x2 - b/W[1]
    return x1, y1, x2, y2

def plot(train_c1_x1, train_c1_x2, train_c2_x1, train_c2_x2, W, b):
    plt.scatter(train_c1_x1, train_c1_x2, color='g')
    plt.scatter(train_c2_x1, train_c2_x2, color='b')
    # Plot two points on the decision boundary Wx+b=0
    x1, y1, x2, y2 = pts_on_the_line(W, b)
    plt.plot([x1, x2], [y1, y2], 'r-')
    # Plot the normal vector in the direction of positive samples.
    ax = plt.axes()
    ax.arrow((x1+x2)/2, (y1+y2)/2, W[0]/10, W[1]/10, head_width=0.5, head_length=0.5, fc='k', ec='k')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def predict(pt, W, b):
    res = W.dot(pt) + b
    if res < 0:
        return -1
    else:
        return 1

def train_perceptron(c1, c2):
    num_iter = 17
    W = np.array([0.1, -0.5])
    b = 0.0
    learninig_rate = 5.0
    errors = 0
    train_c1_x1 = []
    train_c1_x2 = []
    train_c2_x1 = []
    train_c2_x2 = []
    for i in xrange(num_iter):
        if ((i == 0) or (np.log2(i)%1 == 0)):
            print "Iteration", i
            plot(train_c1_x1, train_c1_x2, train_c2_x1, train_c2_x2, W, b)
        # Pick a random point from the training set.
        pt, y = get_train_pt(c1, c2, i%n)
        if (y < 0):
            train_c1_x1.append(pt[0])
            train_c1_x2.append(pt[1])
        else:
            train_c2_x1.append(pt[0])
            train_c2_x2.append(pt[1])
        y_pred = predict(pt, W, b)
        err = y - y_pred
        if (err):
            W += learninig_rate * err * pt
            b += learninig_rate * err * 1

# Run the perceptron algorithm for a few iterations.
train_perceptron(c1, c2)

# Run the perceptron algorithm for a few iterations.
c1_mu = [low, high]
c2_mu = [high, low]
c1, c2 = generate_data(c1_mu, c2_mu, n)
train_perceptron(c1, c2)

c1_mu = [low, low]
c2_mu = [low+1.0, low+1.0]
c1, c2 = generate_data(c1_mu, c2_mu, n)
train_perceptron(c1, c2)







