import numpy as np
import visual_bow as bow
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
import glob
import random
import warnings

SCORING = 'f1_micro'
print 'Scoring grid search with metric: %s' % SCORING

# Get all possible negative images and label them False
positive_folder='panda'
all_negs = [(path, False) for path in bow.neg_img_cal101(positive_folder)]
print '%i total negative imgs to choose from' % len(all_negs)
print all_negs[:5]

# Get all the positive images you have (in the panda_rip folder) and label them True
positive_imgs = [(path, True) for path in glob.glob('panda_rip/*')]
print '%i positive images' % len(positive_imgs)
print positive_imgs[:5]

# take N random negative images, where N is no of positive images
# then concatenate N pos + N neg and shuffle.
chosen_negs = random.sample(all_negs, len(positive_imgs))
imgs = chosen_negs + positive_imgs

np.random.shuffle(imgs)

print '%i total images (1:1 positive:negative)' % len(imgs)
print imgs[:5]

get_ipython().run_cell_magic('time', '', '\nimg_descs, y = bow.gen_sift_features(imgs)')

# joblib.dump(img_descs, 'pickles/img_descs/img_descs.pickle')
# joblib.dump(y, 'pickles/img_descs/y.pickle')

# generate indexes for train/test/val split
training_idxs, test_idxs, val_idxs = bow.train_test_val_split_idxs(
    total_rows=len(imgs), 
    percent_test=0.15, 
    percent_val=0.15
)

get_ipython().run_cell_magic('time', '', "\nK_CLUSTERS = 250\n\n# MiniBatchKMeans annoyingly throws tons of deprecation warnings that fill up the notebook. Ignore them.\nwarnings.filterwarnings('ignore')\n\nX, cluster_model = bow.cluster_features(\n    img_descs, \n    training_idxs=training_idxs, \n    cluster_model=MiniBatchKMeans(n_clusters=K_CLUSTERS)\n)\n\nwarnings.filterwarnings('default')\n\nX_train, X_test, X_val, y_train, y_test, y_val = bow.perform_data_split(X, y, training_idxs, test_idxs, val_idxs)")

# for obj, obj_name in zip( [X_train, X_test, X_val, y_train, y_test, y_val], 
#                          ['X_train', 'X_test', 'X_val', 'y_train', 'y_test', 'y_val'] ):
#     joblib.dump(obj, 'pickles/feature_data/%s.pickle' % obj_name)

# for obj_name in ['X_train', 'X_test', 'X_val', 'y_train', 'y_test', 'y_val']:
#     exec("{obj_name} = joblib.load('pickles/feature_data/{obj_name}.pickle')".format(obj_name=obj_name))
#     exec("print obj_name, len({0})".format(obj_name))

get_ipython().run_cell_magic('time', '', "\n# c_vals = [0.0001, 0.01, 0.1, 1, 10, 100, 1000]\nc_vals = [0.1, 1, 5, 10]\n# c_vals = [1]\n\ngamma_vals = [0.5, 0.1, 0.01, 0.0001, 0.00001]\n# gamma_vals = [0.5, 0.1]\n# gamma_vals = [0.1]\n\nparam_grid = [\n  {'C': c_vals, 'kernel': ['linear']},\n  {'C': c_vals, 'gamma': gamma_vals, 'kernel': ['rbf']},\n ]\n\nsvc = GridSearchCV(SVC(), param_grid, n_jobs=-1, scoring=SCORING)\nsvc.fit(X_train, y_train)\nprint 'train score (%s):'%SCORING, svc.score(X_train, y_train)\nprint 'test score (%s):'%SCORING, svc.score(X_test, y_test)\n\nprint svc.best_estimator_")

for img_path, label in random.sample(all_negs, 10):
    print img_path, svc.predict(bow.img_to_vect(img_path, cluster_model))

# joblib.dump(svc.best_estimator_, 'pickles/svc/svc.pickle')
# joblib.dump(cluster_model, 'pickles/cluster_model/cluster_model.pickle')

get_ipython().run_cell_magic('time', '', "\nada_params = {\n    'n_estimators':[100, 250, 500, 750],\n    'learning_rate':[0.8, 0.9, 1.0, 1.1, 1.2]\n}\n\n# ada = AdaBoostClassifier(n_estimators=MAX_ESTIMATORS, learning_rate=0.8)\nada = GridSearchCV(AdaBoostClassifier(), ada_params, n_jobs=-1, scoring=SCORING)\nada.fit(X_train, y_train)\nprint 'train score (%s):'%SCORING, ada.score(X_train, y_train)\nprint 'test score (%s):'%SCORING, ada.score(X_test, y_test)\nprint ada.best_estimator_")

# joblib.dump(ada.best_estimator_, 'pickles/ada/ada.pickle');
# print 'picked adaboost'

