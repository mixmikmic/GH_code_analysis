def make_it_quack(something_duck_like):
    """
        Take something that can quack, and make it quack!
    """
    something_duck_like.quack()

# define some animals
class Duck(object):
    def quack(self):
        print("Quack quack")

class Ferret(object):
    # ferrets can't normally quack, but this one's cunning
    def quack(self):
        print("Quack quack")

donald = Duck()
fred = Ferret()

print(type(donald))
print(type(fred))

make_it_quack(donald)
make_it_quack(fred)

class MyFakeClassifier():
    def fit(self, x, y):
        print("Working VERY HARD...")
    
    def predict(self, x):
        # predict 0 no matter what
        return [0 for item in range(len(x))]

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = iris.data
y = iris.target

# write a function to give us a train-test accuracy score
def get_accuracy(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy: {}\n{}".format(accuracy_score(y_test, y_pred),
                                    confusion_matrix(y_test, y_pred)))

rf = RandomForestClassifier()
random_model = MyFakeClassifier()

print("Random Forest\n")
get_accuracy(rf, X, y)
print("\nMy Random Estimator\n")
get_accuracy(random_model, X, y)

