import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../BIA660D_Group_1_Project/eda/hoboken_step1.csv')
data['restaurant_rating'].fillna(data['restaurant_rating'].mean(), inplace=True)
data.head(3)

rating_distribution = data['user_rating'].value_counts().loc[list(range(1,6))]
x = rating_distribution.plot.pie(title = 'User Ratings')

# Turn restaurant categories as a list of average ratings 
type_data = pd.DataFrame.from_csv('../BIA660D_Group_1_Project/eda/type_data.csv')
type_data.head(3)

x = type_data.hist()

sorted = type_data.sort_values(by='Average_Score', ascending=False)
best_worst = sorted[:5].append(sorted[-5:])
x = best_worst.plot.bar()









