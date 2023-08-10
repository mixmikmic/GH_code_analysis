# Import the standard packages for doing math operations, data manipulation, and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the IMDB data
movie_data = pd.read_csv("movie_metadata.csv")

# Show all the variables in the data set
movie_data.columns.values

# Drop rows with missing gross revenue data (standard practice in regression analysis)
movie_data = movie_data.dropna(subset = ['gross'])

# Only keep data on movies from the US
movie_data = movie_data[movie_data['country'] == "USA"]

# Assuming a average US inflation rate of 2.5%, we convert gross revenue in terms of 2017 dollars
movie_data[['gross']] = (1.025**(2017-movie_data['title_year']))*movie_data['gross']

# Only keep the variables of interest, 'imdb_score' and 'gross'
movie_data = movie_data[['gross','imdb_score']]

# Let's scale the gross revenue to be in millions of dollars so its easier to read
movie_data[['gross']] = movie_data[['gross']]/1000000

# Summary statistics
movie_data.describe()

# Visualize data
plt.scatter(movie_data['imdb_score'], movie_data['gross'])

# Chart title
plt.title('IMDB Rating and Gross Sales')

# y label
plt.ylabel('Gross sales revenue ($ millions)')

# x label
plt.xlabel('IMDB Rating (0 - 10)')

plt.show()


# Data size, this is derivitive with respect to B0^2
n = len(movie_data.index)

# Sum(IMDB_score), this is derivitive with respect to B0B1 and B1B0
sum_imdb = sum(movie_data['imdb_score'])

# Sum(IMDB_score^2), this is derivitive with respect to B1^2
sum_sq_imdb = sum(np.power(movie_data['imdb_score'],2))

# Initialize hessian matrix as it doesn't depend on linear regression parameters
H = [[n,sum_imdb],[sum_imdb,sum_sq_imdb]]

# Define threshold for convergence, epsilon = 10^-4
epsilon = 10 ** -4

# Criterion to be updated in newton's method
criterion = 10

# Initialize parameters
new_param = np.array((-50,10))

# Define Newton's method update rule as a function
def newton_update(old_param, hessian, jacobian):
    # Matrix version of newton's update
    new_param = np.array(np.subtract(old_param,np.dot(np.linalg.inv(hessian),jacobian)))
    
    return new_param

# Gross revenue np vector
gross_rev = np.array(movie_data['gross'])

# IMDB rating np vector 
imdb_score = np.array(movie_data['imdb_score'])

# Implement Newton's method to find line of best fit (Itterate until convergence)
while criterion > epsilon:
    
    # Reset old parameter (itteration t)
    old_param = new_param
    
    # Compute gradient vector
    J_pos0 = -1*(np.nansum(gross_rev)-n*old_param[0]-old_param[1]*sum_imdb)
    J_pos1 = -1*(np.nansum(gross_rev*imdb_score)-old_param[0]*sum_imdb-old_param[1]*sum_sq_imdb)
    J = np.array([J_pos0,J_pos1])
    
    # Apply newton's update rule (itteration t+1)
    new_param = newton_update(old_param, H, J)
    
    # Compute criterion function using euclidean distance 
    criterion = np.linalg.norm(new_param-old_param)

# Let's see the estimated (Beta0, Beta1) parameters from newton's algorithm
print new_param

# Plot line of best fit to scatter plot above using the estimated parameters
plt.plot(imdb_score, new_param[1]*imdb_score + new_param[0], '-k')

# Visualize data
plt.scatter(imdb_score, gross_rev)

# Chart title
plt.title('IMDB Rating and Gross Sales')

# y label
plt.ylabel('Gross sales revenue ($ millions)')

# x label
plt.xlabel('IMDB Rating (0 - 10)')

plt.show()

