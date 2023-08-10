import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()

df =  pd.DataFrame( data=boston.data )
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df['MEDV'] = boston.target
df.head()

X = df[['RM']].values
y = df[['MEDV']].values # Add more [] to avoid getting WARNING in standardization

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X_std = sc_X.fit_transform(X)
y_std = sc_y.fit_transform(y)[:, 0] # [:, 0] for make N-d list to 1-d list

from AdalineGD import AdalineGD
lr = AdalineGD()
lr.fit(X_std, y_std)

plt.figure(figsize=(8, 6))
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return

plt.figure(figsize=(8,6))
lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of room [RM] (Standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (Standardized)' )
plt.show()

# Transform standardized price to real price
num_room_std = sc_X.transform([[5.0]])
price_std = lr.predict(num_room_std)
print 'Price in $1000\'s: %.3f'%( sc_y.inverse_transform(price_std) )

print 'Slope: %.3f'% lr.w_[1]
print 'Intercept: %.3f'% lr.w_[0] # since standarized, the intercept should be 0

