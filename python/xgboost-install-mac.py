get_ipython().system('pip3 show xgboost')

import pip
pip.main(['install', 'xgboost'])

import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=15, nthread=-1, seed=1111)
model



