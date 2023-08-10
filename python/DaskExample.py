from sklearn.model_selection import train_test_split
# Split the training data into training and test sets for cross-validation
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=42)

from dask import compute, delayed

# Define process to be executed
def process(Xin):
    return model.predict(Xin)

# Create dask objects
# Reshape is necessary because the format of x as drawm from Xtest 
# is not what sklearn wants.
dobjs = [delayed(process)(x.reshape(1,-1)) for x in Xtest]

# Do the computation, one process at a time instead of all at once
import dask.threaded
ypred = compute(*dobjs, get=dask.threaded.get)

# The dask output is sort of odd, so this is just to put the result back into the expected format.
ypred = np.array(ypred).reshape(1,-1)[0]

fig = plt.figure(figsize=(6,6))
plt.scatter(ytest,ypred)
plt.show()

