from IPython.display import Image
get_ipython().magic('matplotlib inline')

Image(filename= 'C:/PythonProjects/NaiveBayes/gaussian.png', width='600',height='800')

Image(filename= 'C:/PythonProjects/NaiveBayes/wiki-example.png',width='500',height='500')

Image(filename= 'C:/PythonProjects/NaiveBayes/gaussianlikelihood.png',width='',height='1200')

# Import Dependencies
import numpy as np
import pandas as pd

# Import Dataset
df = pd.read_csv('dataset/data_banknote_authentication.csv')

# Have a look at the dataset
df.head()

# Adding names to Columns
df.columns = ['variance','skewness','curtosis','entropy','class']

df.head()

# Describe the Data
df.describe()

# Count the number of Original and Fake Notes
# Original: Class = 1
# Fake: Class = 0

num_classOriginal = df['class'][df['class'] == 1].count()
num_classFake = df['class'][df['class'] == 0].count()
total = len(df)

print('Number of Original Bank Notes: ',num_classOriginal)
print('Number of Fake Bank Notes: ',num_classFake)
print('Total number of Notes: ',total)

# Calculating the Prior Probabilities

# Probability(Original Note)
Probb_Original = num_classOriginal/total
print('Probability of Original Notes in Dataset: ',Probb_Original)

# Probability(Fake Note)
Probb_Fake = num_classFake/total
print('Probability of Fake Notes in Dataset: ',Probb_Fake)

# Data Mean
data_mean = df.groupby('class').mean()
print('Mean: \n',data_mean)

print('\n')

# Data Variance
data_variance = df.groupby('class').var()
print('Variance: \n',data_variance)

# Function to Calculate Likelihood Probability
def p_x_given_y(x, mean_y, variance_y):
    probb = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))
    return probb

# Testing Data
# Originally, this data represents Fake Bank Note
a = [3.2032,5.7588,-0.75345,-0.61251]

# Probability the Notes are Original
prob_orig = [Probb_Original * p_x_given_y(a[0], data_mean['variance'][1], data_variance['variance'][1]) * p_x_given_y(a[1], data_mean['skewness'][1], data_variance['skewness'][1]) * p_x_given_y(a[2], data_mean['curtosis'][1], data_variance['curtosis'][1]) * p_x_given_y(a[3], data_mean['entropy'][1], data_variance['entropy'][1])]

# Probability the Notes are Fake
prob_fake = [Probb_Fake * p_x_given_y(a[0], data_mean['variance'][0], data_variance['variance'][0]) * p_x_given_y(a[1], data_mean['skewness'][0], data_variance['skewness'][0]) * p_x_given_y(a[2], data_mean['curtosis'][1], data_variance['curtosis'][0]) * p_x_given_y(a[3], data_mean['entropy'][0], data_variance['entropy'][0])]

# Testing the Classifier
if (prob_orig > prob_fake):
    print('Congratulations !! Your Bank Note is Original...')
else:
    print('Sorry !! Your Bank Note is a Fake !!')

