import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

x = np.arange(100)
y = 10 + 5 * x
plt.plot(x, y)
plt.title("True function of y")
plt.show()

iters = 50000
betas = np.zeros((iters, 2))
vars = np.zeros((iters, 2))
c = np.array([[1, 0], [0, 1]])
X = np.hstack((np.ones((100, 1)), x[:, np.newaxis]))
    
for i in range(iters):
    y2 = y + np.random.normal(0, 1, 100) * 100
    betas[i, :] = np.linalg.lstsq(X, y2)[0]
    y_hat = X.dot(betas[-1, :])
    sigma_hat = np.sum((y_hat - y2) ** 2) / (X.shape[0] - X.shape[1])
    desvar = c.dot(np.linalg.pinv(X.T.dot(X))).dot(c.T)
    vars[i, :] = np.diag(sigma_hat * desvar)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.hist(betas[:, 0], bins=50)
plt.title(r"$\hat{\beta}_{0}$", fontsize=20)

plt.subplot(1, 2, 2)
plt.hist(betas[:, 1], bins=50)
plt.title(r"$\hat{\beta}_{x}$", fontsize=20)

plt.show()

betas.mean(axis=0)

betas.var(axis=0)

vars.mean(axis=0)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.hist(vars[:, 0], bins=50)
plt.title(r"$\hat{\mathrm{var}[\beta}_{0}]$", fontsize=20)

plt.subplot(1, 2, 2)
plt.hist(vars[:, 1], bins=50)
plt.title(r"$\hat{\mathrm{var}[\beta}_{x}]$", fontsize=20)

plt.show()

