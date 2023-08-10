import numpy as np
import matplotlib.pyplot as plt

probs = np.array([x for x in range(1,10)])/10
loss = np.log(probs)
plt.plot(probs, loss)

classes = ['cat', 'dog', 'bird']
y = [1,0,0]
y_hat = [0.4, 0.3, 0.3]

l1 = -np.sum(y * np.log(y_hat))

y = [0,0,1]
y_hat = [0.1, 0.1, 0.8]

l2 = -np.sum(y * np.log(y_hat))

y = [0,1,0]
y_hat = [0.4, 0.3, 0.3]

l3 = -np.sum(y * np.log(y_hat))

l1, l2, l3

np.mean([l1,l2,l3])

probs = np.array([x for x in range(1,10)])/10







