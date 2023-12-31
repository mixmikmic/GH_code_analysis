import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib
get_ipython().magic('matplotlib inline')

# To change the default figure size
matplotlib.rcParams['figure.figsize'] = [20.0, 10.0]

after_feature_selection_summer = pd.read_csv('data/after_feature_selection_summer.csv')
after_feature_selection_rainy = pd.read_csv('data/after_feature_selection_rainy.csv')
after_feature_selection_winter = pd.read_csv('data/after_feature_selection_winter.csv')
years_1901_1980 = list(range(1901, 1981))
years_1981_2000 = list(range(1981, 2001))
years_2002_2020 = list(range(2002, 2021))
all_years = years_1901_1980 + years_1981_2000

def plotGraph(labels, X, y, title, model):
    matplotlib.pyplot.scatter(labels, y, color = 'black', label = 'Data')
    matplotlib.pyplot.plot(labels, model.predict(X.as_matrix()), color = 'cornflowerblue', linewidth = 2, label = 'linear_reg_model')
    matplotlib.pyplot.xlabel('Year')
    matplotlib.pyplot.ylabel('Rainfall')
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.legend() 

# after_feature_selection_summer.head()
# after_feature_selection_summer.loc[:, [True, True, True, False]].head()
# after_feature_selection_summer.iloc[:, :3].head()
# after_feature_selection_summer.loc[:, :"summer_wetDayFrequency_data"].head()

X_train, X_test, y_train, y_test = train_test_split(after_feature_selection_summer.iloc[:, :3], 
                                                    after_feature_selection_summer.summer_rainfall_data, 
                                                    test_size=0.2, 
                                                    random_state=0)

linear_regression_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
linear_regression_model.fit(X_train.as_matrix(), y_train.as_matrix())
plotGraph(labels = years_1981_2000, 
          X = X_test, 
          y = y_test, 
          title = 'Rainfall Prediction for Summer', 
          model = linear_regression_model)

print("score: ", linear_regression_model.score(X_test.as_matrix(), y_test.as_matrix()))

type(after_feature_selection_summer.summer_rainfall_data.as_matrix())
pd.DataFrame({"years": np.asarray(all_years)}).as_matrix();
after_feature_selection_summer.columns

matplotlib.pyplot.scatter(years_1981_2000, y_test, color = 'black', label = 'Data')
matplotlib.pyplot.xlabel('Year')
matplotlib.pyplot.ylabel('Rainfall')
# matplotlib.pyplot.title(title)
matplotlib.pyplot.plot(years_1901_1980, y_train, color = 'cornflowerblue', linewidth = 1, label = 'Rainfall trend')
matplotlib.pyplot.plot(years_1981_2000, linear_regression_model.predict(X_test.as_matrix()), color = 'yellow', linewidth = 2, label = 'Rainfall prediction')

linear_regression_model_trend = make_pipeline(PolynomialFeatures(3), LinearRegression())
linear_regression_model_trend.fit(pd.DataFrame({"years": np.asarray(all_years)}).as_matrix(),after_feature_selection_summer.summer_rainfall_data.as_matrix())
matplotlib.pyplot.plot(all_years, linear_regression_model_trend.predict(pd.DataFrame({"years": np.asarray(all_years)}).as_matrix()), color = 'magenta', linewidth = 1, label = 'Rainfall predicted trend')
matplotlib.pyplot.plot(years_2002_2020, linear_regression_model_trend.predict(pd.DataFrame({"years": np.asarray(years_2002_2020)}).as_matrix()), color = 'magenta', linewidth = 1, label = 'Rainfall predicted trend')

# summer_precipitation_data
linear_regression_model_trend.fit(pd.DataFrame({"years": np.asarray(all_years)}).as_matrix(),after_feature_selection_summer.summer_precipitation_data.as_matrix())
matplotlib.pyplot.plot(all_years, linear_regression_model_trend.predict(pd.DataFrame({"years": np.asarray(all_years)}).as_matrix()), color = 'black', linewidth = 1, label = 'Precipitation predicted trend')
predicted_precipitation_data = linear_regression_model_trend.predict(pd.DataFrame({"years": np.asarray(years_2002_2020)}).as_matrix())
matplotlib.pyplot.plot(years_2002_2020, predicted_precipitation_data, color = 'black', linewidth = 1, label = 'Precipitation predicted trend')

# summer_wetDayFrequency_data
linear_regression_model_trend.fit(pd.DataFrame({"years": np.asarray(all_years)}).as_matrix(),after_feature_selection_summer.summer_wetDayFrequency_data.as_matrix())
matplotlib.pyplot.plot(all_years, linear_regression_model_trend.predict(pd.DataFrame({"years": np.asarray(all_years)}).as_matrix()), color = 'purple', linewidth = 1, label = 'Wet Day frequency predicted trend')
predicted_wetDayFreq_data = linear_regression_model_trend.predict(pd.DataFrame({"years": np.asarray(years_2002_2020)}).as_matrix())
matplotlib.pyplot.plot(years_2002_2020, predicted_wetDayFreq_data, color = 'purple', linewidth = 1, label = 'Wet Day frequency predicted trend')
matplotlib.pyplot.legend() 

# summer_cloudcover_data
linear_regression_model_trend.fit(pd.DataFrame({"years": np.asarray(all_years)}).as_matrix(),after_feature_selection_summer.summer_cloudcover_data.as_matrix())
matplotlib.pyplot.plot(all_years, linear_regression_model_trend.predict(pd.DataFrame({"years": np.asarray(all_years)}).as_matrix()), color = 'green', linewidth = 1, label = 'Cloud Cover predicted trend')
predicted_cloudCover_data = linear_regression_model_trend.predict(pd.DataFrame({"years": np.asarray(years_2002_2020)}).as_matrix())
matplotlib.pyplot.plot(years_2002_2020, predicted_cloudCover_data, color = 'green', linewidth = 1, label = 'Cloud Cover predicted trend')

# predicted data set.
predicted_data_set = pd.DataFrame({"predicted_precipitation": predicted_precipitation_data,
                                  "predicted_wetDayFreq": predicted_wetDayFreq_data,
                                  "predicted_cloudCover": predicted_cloudCover_data})
matplotlib.pyplot.plot(years_2002_2020, linear_regression_model.predict(predicted_data_set.as_matrix()), color = 'yellow', linewidth = 2, label = 'linear_reg_model_predicted')
matplotlib.pyplot.legend() 

