import visual_bow as bow
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

positive_folder = 'panda'
imgs = bow.binary_labeled_img_from_cal101(positive_folder)

X_train, X_test, y_train, y_test, kmeans = bow.gen_bow_features(imgs, test_train_ratio=0.8, K_clusters=750)

c_vals = [0.0001, 0.01, 0.1, 1, 5, 10, 100, 1000]

param_grid = [
  {'C': c_vals, 'kernel': ['linear']},
  {'C': c_vals, 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf']},
 ]

svc = GridSearchCV(SVC(), param_grid, n_jobs=-1)
svc.fit(X_train, y_train)
svc.score(X_test, y_test)

svc.best_estimator_

for img_path in ['kanye_glasses.jpeg', 
                 'kanye_glasses2.jpeg', 
                 'more_pandas/0001.jpeg', 
                 '101_ObjectCategories/brontosaurus/image_0001.jpg',
                 '101_ObjectCategories/brontosaurus/image_0002.jpg',
                 '101_ObjectCategories/dalmatian/image_0001.jpg',
                 '101_ObjectCategories/dalmatian/image_0002.jpg'
                ]:
    print img_path, svc.predict(bow.img_to_vect(img_path, kmeans))

# Uncomment to pickle the best SVC classifier
###########
# from sklearn.externals import joblib
# joblib.dump(svc.best_estimator_, 'pickles/panda/panda_svc.pickle')



