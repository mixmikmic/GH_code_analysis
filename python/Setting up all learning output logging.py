import os

from keras.utils import to_categorical
from data import load_data
from results import reprocess_dir
from data import load_data

data_folder = 'ModelNet10'
(_, _), (x_test, y_test), target_names = load_data(data_folder)
y_test = to_categorical(y_test)

NAME = 'results/'
one_dir = os.listdir(NAME)[0]

model_folder = os.listdir(os.path.join(NAME, one_dir))

os.listdir(os.path.join(NAME, os.listdir(NAME)[0]))

os.path.isfile(os.path.join(NAME, one_dir, model_folder[0]))


data_folder = 'ModelNet10'
NAME = 'results/'

one_dir = os.listdir(NAME)[0]

model_folder = os.listdir(os.path.join(NAME, one_dir))

os.listdir(os.path.join(NAME, os.listdir(NAME)[0]))

os.path.isfile(os.path.join(NAME, one_dir, model_folder[0]))

(_, _), (x_test, y_test), target_names = load_data(data_folder)
# x_train, y_train, x_val, y_val = stratified_shuffle(x_train, y_train, test_size=.1)
# x_train, y_train = upsample_classes(x_train, y_train)
# y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# y_val = to_categorical(y_val)



reprocess_dir(os.path.join(NAME, one_dir), x_test, y_test, target_names)

print(os.path.join(NAME, one_dir))

folders = []
root = 'results'
for sub_dir in os.listdir(root):
    print(os.path.join(root, sub_dir))
    folders.append(os.path.join(root, sub_dir))

list(filter(lambda x: 'ModelNet40' in x, folders))

data_folder = 'ModelNet40'
(_, _), (x_test, y_test), target_names = load_data(data_folder)
y_test = to_categorical(y_test)

model40_folders = [i for i in folders if '40' in i]

model40_folders

reprocess_dir(model40_folders[1], x_test, y_test, target_names)



























