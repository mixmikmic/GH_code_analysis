import graphlab as gl

data = gl.SFrame('fraud_detection.sf')

data.head(3)

len(data)

data.show()

# Tell GraphLab to display canvas in the notebook itself
gl.canvas.set_target('ipynb')

data.show(view='BoxWhisker Plot', x='fraud', y='payment amount')

# Transform string date into datetime type.
# This will help us further along to compare dates.
data['transaction date'] = data['transaction date'].str_to_datetime(str_format='%d.%m.%Y')

# Split date into its components and set them as categorical features 
data.add_columns(data['transaction date'].split_datetime(limit=['year','month','day'], column_name_prefix='transaction'))
data['transaction.year'] = data['transaction.year'].astype(str)
data['transaction.month'] = data['transaction.month'].astype(str)
data['transaction.day'] = data['transaction.day'].astype(str)

# Create day of week feature and set it as a categorical feature
data['transaction week day'] = data['transaction date'].apply(lambda x: x.weekday())
data['transaction week day'] = data['transaction week day'].astype(str)

data.head(3)

# Create new features and transform them into true/false indicators
data['same country'] = (data['customer country'] == data['business country']).astype(str)
data['same person'] = (data['customer'] == data['cardholder']).astype(str)
data['expiration near'] = (data['credit card expiration year'] == data['transaction.year']).astype(str)

counts = data.groupby('transaction id', {'unique cards per transaction' : gl.aggregate.COUNT_DISTINCT('credit card number'),
                                         'unique cardholders per transaction' : gl.aggregate.COUNT_DISTINCT('cardholder'),
                                         'tries per transaction' : gl.aggregate.COUNT()})
counts.head(3)

counts.show()

data = data.join(counts)

data.show(view='BoxWhisker Plot', x='fraud', y='unique cards per transaction')

print 'Number of columns', len(data.column_names())

from datetime import datetime

split = data['transaction date'] > datetime(2015, 6, 1)
data.remove_column('transaction date')

train = data[split == 0]
test = data[split == 1]

print 'Training set fraud'
train['fraud'].show()

print 'Test set fraud'
test['fraud'].show()

logreg_model = gl.logistic_classifier.create(train,
                                             target='fraud',
                                             validation_set=None)

print 'Logistic Regression Accuracy', logreg_model.evaluate(test)['accuracy']
print 'Logistic Regression Confusion Matrix\n', logreg_model.evaluate(test)['confusion_matrix']

boosted_trees_model = gl.boosted_trees_classifier.create(train, 
                                                         target='fraud',
                                                         validation_set=None)

print 'Boosted trees Accuracy', boosted_trees_model.evaluate(test)['accuracy']
print 'Boosted trees Confusion Matrix\n', boosted_trees_model.evaluate(test)['confusion_matrix']

boosted_trees_model = gl.boosted_trees_classifier.create(train, 
                                                         target='fraud',
                                                         validation_set=None,
                                                         max_iterations=40,
                                                         max_depth=9,
                                                         class_weights='auto')

print 'Boosted trees Accuracy', boosted_trees_model.evaluate(test)['accuracy']
print 'Boosted trees Confusion Matrix\n', boosted_trees_model.evaluate(test)['confusion_matrix']

# Inspect the features most used by the boosted trees model
boosted_trees_model.get_feature_importance()

state_path = 's3://gl-demo-usw2/predictive_service/demolab/ps-1.8.5'

ps = gl.deploy.predictive_service.load(state_path)

# Pickle and send the model over to the server.
ps.add('fraud', boosted_trees_model)
ps.apply_changes()

# Predictive services must be displayed in a browser
gl.canvas.set_target('browser')

ps.show()

ps.query('fraud', method='predict', data={'dataset' : test[0]})

test[0]['fraud']

