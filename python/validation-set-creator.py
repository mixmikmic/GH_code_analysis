import os as os
from os import *
import random

competition_name = 'kaggle_plant_seedlings'

PATH = os.path.join('..', competition_name, 'data', 'sample') + '/'
#PATH = os.path.join('..', competition_name, 'data') + '/'

classes = get_ipython().getoutput('ls {PATH}/train')
print(classes)

def copy_to_valid(chosen_files, class_dir):
    for i in range (len(chosen_files)):
        get_ipython().system('mv "{PATH}/train/{class_dir}/{chosen_files[i]}" "{PATH}/valid/{class_dir}"')

validation_ratio = 0.2
for classname in classes:
    os.makedirs(f'{PATH}/valid/{classname}', exist_ok=True)
    list_of_files = get_ipython().getoutput('ls "{PATH}/train/{classname}"')
    random.shuffle(list_of_files)
    n_files_moved=int(len(list_of_files) * validation_ratio)
    selected_files = [list_of_files[m] for m in range(n_files_moved)]
    copy_to_valid(selected_files,classname)



