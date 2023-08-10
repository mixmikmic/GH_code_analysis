from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn import datasets

splits = 3

X = range(12)
y = [0] * 6 + [1] * 6

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

X_train

print("TRAIN:", train_index, "TEST:", test_index)

print(X, y)
print(X_train, y_train)
print(X_test, y_test)

print("KFold")
kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
for train_index, test_index in kfold.split(tx, ty):
    print("TRAIN:", train_index, "TEST:", test_index)

print("\nShuffle Split")
shufflesplit = StratifiedShuffleSplit(n_splits=splits, test_size=1/3, random_state=42)
for train_index, test_index in shufflesplit.split(tx, ty):
    print("TRAIN:", train_index, "TEST:", test_index)



