from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 读取数据集
X, y = load_boston(True)

# 使用sklearn的库分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# 可以看出来训练集有404条数据，13个变量，测试集有102条数据
# 目标变量是一维的
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# sklearn的线性回归对象
model = LinearRegression()

# 拟合训练集
model.fit(X_train, y_train)

# 训练集预测结果
pred_train = model.predict(X_train)

# 测试集预测结果
pred_test = model.predict(X_test)

# 训练集的 Mean Squared Error
print('mse of train:', mean_squared_error(y_train, pred_train))

# 测试集的 Mean Squared Error
print('mse of test:', mean_squared_error(y_test, pred_test))



