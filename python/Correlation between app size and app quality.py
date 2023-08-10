import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

app = pd.read_pickle('/Users/krystal/Desktop/app_cleaned.pickle')
app.head()

app = app.drop_duplicates()

for i in range(0,len(app)):
    unit = app['size'][i][-2:]
    if unit == 'GB':
        app['size'][i] = float(app['size'][i][:-3])*1000
    else:
        app['size'][i] = float(app['size'][i][:-3])

rating_df = app[["name","size","overall_rating", "current_rating", 'num_current_rating', "num_overall_rating"]].dropna()

rating_cleaned = {'1 star':1, "1 and a half stars": 1.5, '2 stars': 2, '2 and a half stars':2.5, "3 stars":3, "3 and a half stars":3.5, "4 stars": 4,
                 '4 and a half stars': 4.5, "5 stars": 5}

rating_df.overall_rating = rating_df.overall_rating.replace(rating_cleaned)

rating_df['weighted_rating'] = np.divide(rating_df['num_current_rating'],rating_df['num_overall_rating'])*rating_df['current_rating']+(1-np.divide(rating_df['num_current_rating'],rating_df['num_overall_rating']))*rating_df['overall_rating']

plt.scatter(rating_df['size'], rating_df['weighted_rating'])
plt.xlabel('Size of app')
plt.ylabel('Quality of app')
plt.title('Relationship between app size and quality')
plt.show()

rating_df_2 = rating_df[rating_df['size'] <= 500]

plt.scatter(rating_df_2['size'], rating_df_2['weighted_rating'])
plt.xlabel('Size of app')
plt.ylabel('Quality of app')
plt.title('Relationship between app size(less than 500) and quality')
plt.show()



