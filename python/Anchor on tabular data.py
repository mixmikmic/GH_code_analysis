from __future__ import print_function
import numpy as np
np.random.seed(1)
import sys
import sklearn
import sklearn.ensemble
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from anchor import utils
from anchor import anchor_tabular

# make sure you have adult/adult.data inside dataset_folder
dataset_folder = '/Users/marcotcr/phd/datasets/'
dataset = utils.load_dataset('adult', balance=True, dataset_folder=dataset_folder)

explainer = anchor_tabular.AnchorTabularExplainer(dataset.class_names, dataset.feature_names, dataset.data, dataset.categorical_names)
explainer.fit(dataset.train, dataset.labels_train, dataset.validation, dataset.labels_validation)

c = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
c.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)
predict_fn = lambda x: c.predict(explainer.encoder.transform(x))
print('Train', sklearn.metrics.accuracy_score(dataset.labels_train, predict_fn(dataset.train)))
print('Test', sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test)))

idx = 0
np.random.seed(1)
print('Prediction: ', explainer.class_names[predict_fn(dataset.test[idx].reshape(1, -1))[0]])
exp = explainer.explain_instance(dataset.test[idx], c.predict, threshold=0.95)

print('Anchor: %s' % (' AND '.join(exp.names())))
print('Precision: %.2f' % exp.precision())
print('Coverage: %.2f' % exp.coverage())

# Get test examples where the anchora pplies
fit_anchor = np.where(np.all(dataset.test[:, exp.features()] == dataset.test[idx][exp.features()], axis=1))[0]
print('Anchor test coverage: %.2f' % (fit_anchor.shape[0] / float(dataset.test.shape[0])))
print('Anchor test precision: %.2f' % (np.mean(predict_fn(dataset.test[fit_anchor]) == predict_fn(dataset.test[idx].reshape(1, -1)))))

print('Partial anchor: %s' % (' AND '.join(exp.names(1))))
print('Partial precision: %.2f' % exp.precision(1))
print('Partial coverage: %.2f' % exp.coverage(1))

fit_partial = np.where(np.all(dataset.test[:, exp.features(1)] == dataset.test[idx][exp.features(1)], axis=1))[0]
print('Partial anchor test precision: %.2f' % (np.mean(predict_fn(dataset.test[fit_partial]) == predict_fn(dataset.test[idx].reshape(1, -1)))))
print('Partial anchor test coverage: %.2f' % (fit_partial.shape[0] / float(dataset.test.shape[0])))

exp.show_in_notebook()

