import datetime
import sys

sys.path.append('../code/')

from rbf_kernel import RBFKernel
from scipy.spatial.distance import squareform

from mnist_helpers import mnist_training, mnist_testing

X_train, y_train = mnist_training()

k = RBFKernel(X_train)

now = datetime.datetime.now()
print("Current date and time using strftime:")
print(now.strftime("%Y-%m-%d %H:%M:%S"))

X = k.transform(X_train)

now = datetime.datetime.now()
print("Current date and time using strftime:")
print(now.strftime("%Y-%m-%d %H:%M:%S"))

x = np.array()



