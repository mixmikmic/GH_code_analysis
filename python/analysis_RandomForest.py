import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [8, 5]

data = pd.read_csv('../output/merged.csv', index_col=0)

#Feature engineering
normList = ['num_complaints',
            'num_food_est',
            'num_food_poi',
            'num_smoking',
            'num_violations',
            'num_Cviolations',
            'num_amer',
            'num_chi',
            'num_jap',
            'num_cafe',
            'num_ita',
            'num_latin',
            'num_mex',
            'num_pizza',
            'num_span']

for i in normList:
    data[i] /= data['num_restaurants']

#Randomizing and splitting dataset
data = data.reindex(np.random.permutation(data.index))
train = data[0:160]
test = data[160::]

features = data.columns[:-1].tolist()
X,y = train[features],train['avg_score']
X_test,y_test = test[features],test['avg_score']

#Setting up Random Forest cross validation
for i in range(10,101,5):
    estimator = RandomForestRegressor(random_state=0, n_estimators=i)
    score = np.mean(cross_val_score(estimator, X, y))
    print "CV score for %d estimators: %.4f" % (i,score)

chosenModel = RandomForestRegressor(random_state=0, n_estimators=35).fit(X,y)
print 'R2 value of the test set using 35 estimators: %.4f' % chosenModel.score(X_test,y_test)

X_whole,y_whole = data[features],data['avg_score']
X_preds = chosenModel.predict(X_whole)
df_model = pd.DataFrame(data=data, index=data.index)
df_model['Predictions'] = X_preds
df_model['Dev_SqDists'] = (X_preds-df_model['avg_score'])**2
df_model = df_model.sort_index()
df_model.to_csv('../output/RandomForest.csv')
df_model.head()

print 'R2 value of the whole set using 35 estimators: %.4f' % chosenModel.score(X_whole,y_whole)

plt.scatter(chosenModel.predict(X_whole),y_whole)
plt.plot(y_whole,y_whole,'r',linewidth=1.5)
plt.title('Prediction by Random Forest (35 estimators): R2=%.2f' % chosenModel.score(X_whole,y_whole))
plt.xlabel('Predicted average score in each zip code')
plt.ylabel('Actual average score in each zip code')
plt.savefig('../figures/RandomForest_35.png')
plt.show()

plt.hist(df_model.Predictions.values-df_model.avg_score.values,10)
plt.title('Histogram of the deviated distances (Random Forest)')
plt.savefig('../figures/RandomForest_hist.png')
plt.show()



