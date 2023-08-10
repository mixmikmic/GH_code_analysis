from data_load import Data_Loader
from predictiv_learning_sklearn import Predictiv_Learner_Sklearn
import pandas as pd

data_loader = Data_Loader()
X_train, y_train, X_test, y_test = data_loader.load_data()
print('data load finished')

X_test.head()

pred_sklear = Predictiv_Learner_Sklearn()
ranked_result = pred_sklear.do_prediction(X_train, X_test, y_train, y_test)
print('ranking done')

#Output the ranked lists for ranklib evaluation, no need for features here
def f(x):
    if x.name == 'qid':
        return 'qid:' + x.astype(str)
    else:
        return x

(ranked_result.apply(lambda x: f(x))[['rel','qid']]
  .to_csv('pointwise_ranked.csv', sep=' ', index=False, header=None))



