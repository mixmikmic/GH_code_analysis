get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd

from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score

data = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])
y_or = data.any(axis=1).astype(np.int)
y_and = data.all(axis=1).astype(np.int)
y_nand = (~data.all(axis=1)).astype(np.int)
y_xor = (data[:, 0] != data[:, 1]).astype(np.int)
df_true = pd.DataFrame({'left': data[:, 0], 'rigth': data[:, 1],
                        'OR': y_or, 'AND': y_and, 'XOR': y_xor, 'NAND': y_nand},
                       columns=['left', 'rigth', 'OR', 'XOR', 'AND', 'NAND'])
df_true

gates = {'or': y_or, 'and': y_and, 'xor': y_xor, 'nand': y_nand}
test = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1],
    [1, 0], [0, 1], [1, 1], [0, 0],
    [1, 1], [1, 0], [0, 1], [0, 0],
])
test_y = {
    'or': test.any(axis=1).astype(np.int),
    'and': test.all(axis=1).astype(np.int),
    'xor': (test[:, 0] != test[:, 1]).astype(np.int),
    'nand': (~test.all(axis=1)).astype(np.int)
}
pcpt = Perceptron(max_iter=10)
for g in ['or', 'and', 'nand', 'xor']:
    pcpt.fit(data, gates[g])
    y_pred = pcpt.predict(test)
    f1 = f1_score(test_y[g], y_pred)
    print('###### Gate:', g.upper(), 'F1:', f1)

x1 = np.linspace(-2, 0, 10)
x2 = np.linspace(0, 2, 10)
y1 = x1*0 - 1
y2 = x2*0 + 1
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(np.hstack([x1, x2]), np.hstack([y1, y2]), color='crimson');

pcpt_or = Perceptron(max_iter=1e6)
pcpt_or.fit(data, y_or)
pcpt_nand = Perceptron(max_iter=1e6)
pcpt_nand.fit(data, y_nand)
pcpt_and = Perceptron(max_iter=1e6)
pcpt_and.fit(data, y_and)


def predict_xor(x):
    left = pcpt_or.predict(x)
    right = pcpt_nand.predict(x)
    return pcpt_and.predict(np.array(list(zip(left, right))))


y_pred = predict_xor(test)
f1_score(test_y['xor'], y_pred)

