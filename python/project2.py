get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.linalg import inv, pinv

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)

data = np.loadtxt('whData.dat',
                  dtype=np.object,
                  comments='#',
                  delimiter=None)

W = data[:,0].astype('float32')
H = data[:,1].astype('float32')

w_mask = ((W > 0) * 1).nonzero()
pred_mask = ((W < 0) * 1).nonzero()


H_to_predict = H[pred_mask]

W = W[w_mask]
H = H[w_mask]

# to make the code more numerically stable we down-scale it by some 
# large number so that all values are between 0 and 1
scale = 200
W = W/scale
H = H/scale
H_to_predict = np.sort(H_to_predict/scale)

# plot the outliers
for pred in H_to_predict:
    ax.plot([pred, pred], [-10, 10], '--', color='black')

W = W.astype('float64')
H = H.astype('float64')

# solve the solution "zufuss"
# ------------ d = 1 ----------
X = np.vander(H, 1+1)  # (21,2)
y = np.array([W]).T    # (21,1)
w = inv(X.T @ X) @ (X.T @ y)
incl, intercpt = w
ax.plot([0, 200], [incl * 0 + intercpt, incl * 200 + intercpt], 
         label='d = 1')

solve_d1 = lambda x: incl * x + intercpt

# ------------ d = 5 ----------
d = 5
X = np.vander(H, d+1)
y = np.array([W]).T        # (21,1)

# pinv is more numerically stable than the 'naive' implementation above
w = np.squeeze(pinv(X) @ y)  # pinv = (X^T X) X^T
_x_ = np.linspace( 0, 1, 10000)
p_d5 = np.poly1d(w)
ax.plot(_x_, p_d5(_x_), label="d = 5")

# ------------ d = 10 ----------
d = 10
X = np.vander(H, d+1)
y = np.array([W]).T        # (21,1)

# pinv is more numerically stable than the 'naive' implementation above
w = np.squeeze(pinv(X) @ y)  # pinv = (X^T X) X^T
_x_ = np.linspace( 0, 1, 10000)
p_d10 = np.poly1d(w)
ax.plot(_x_, p_d10(_x_), label="d = 10")

# ------------------------------

ax.scatter(H, W, color='black')
ax.set_xlabel('Height')
ax.set_ylabel('Weight')

ax.set_ylim([0/scale, 120/scale])
ax.set_xlim([155/scale, 190/scale])
ax.set_xticklabels([int(x * scale) for x in ax.get_xticks()])
ax.set_yticklabels([int(y * scale) for y in ax.get_yticks()])

print("d=5")
print("\t",H_to_predict*scale)
print("\t",p_d5(H_to_predict)*scale)

print("d=10")
print("\t",H_to_predict*scale)
print("\t",p_d10(H_to_predict)*scale)

ax.scatter(H_to_predict, solve_d1(H_to_predict), color='blue', s=80)
ax.scatter(H_to_predict, p_d5(H_to_predict), color='orange', s=80)
ax.scatter(H_to_predict, p_d10(H_to_predict), color='green', s=80)

plt.legend()
plt.show()

