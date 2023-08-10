import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# Note that there are no NANs in these data; '?' is
# used when there is missing information
accepts = pd.read_csv('data/chefmozaccepts.csv')
cuisine = pd.read_csv('data/chefmozcuisine.csv')
hours = pd.read_csv('data/chefmozhours4.csv')
parking = pd.read_csv('data/chefmozparking.csv')
geo = pd.read_csv('data/geoplaces2.csv') 
usercuisine = pd.read_csv('data/usercuisine.csv')
payment = pd.read_csv('data/userpayment.csv')
profile = pd.read_csv('data/userprofile.csv')
rating = pd.read_csv('data/rating_final.csv')

accepts.head()

print("There are {} unique placeID's.".format(len(accepts.placeID.unique())))
print("There are {} unique Rpayment categories:".format(len(accepts.Rpayment.unique())))
print(accepts.Rpayment.unique())

cuisine.head()

print("There are {} unique placeID's.".format(len(cuisine.placeID.unique())))
print("There are {} unique Rcuisine categories:".format(len(cuisine.Rcuisine.unique())))
print(cuisine.Rcuisine.unique())

hours.head()

print("There are {} unique placeID's.".format(len(hours.placeID.unique())))

parking.head()

print("There are {} unique placeID's.".format(len(parking.placeID.unique())))
print("There are {} unique parking_lot categories:".format(len(parking.parking_lot.unique())))
print(parking.parking_lot.unique())

geo.head()

print("There are {} unique placeID's.".format(len(geo.placeID.unique())))

usercuisine.head()

print("There are {} unique userID's.".format(len(usercuisine.userID.unique())))
print("There are {} unique Rcuisine categories:".format(len(usercuisine.Rcuisine.unique())))
print(usercuisine.Rcuisine.unique())

payment.head()

print("There are {} unique userID's.".format(len(payment.userID.unique())))
print("There are {} unique Upayment categories:".format(len(payment.Upayment.unique())))
print(payment.Upayment.unique())

profile.head()

print("There are {} unique userID's.".format(len(profile.userID.unique())))

rating.head()

print("There are {} unique userID's.".format(len(rating.userID.unique())))
print("There are {} unique placeID's.".format(len(rating.placeID.unique())))
print("There are {} * 3 ratings.".format(len(rating)))

rating.iloc[:,2:].describe()

res_all = np.concatenate((accepts.placeID, cuisine.placeID, 
                          hours.placeID, parking.placeID, geo.placeID))
res_all = np.sort( np.unique(res_all) ) # All the placeID's

print("There are {} restaurants.".format(len(res_all)))

user_all = np.concatenate((usercuisine.userID, payment.userID, profile.userID))
user_all = np.sort( np.unique(user_all) ) # All the userID's

print("There are {} users.".format(len(user_all)))

overall_rating = pd.DataFrame( np.zeros((len(res_all),len(user_all)))-1.0, 
                              columns=user_all, index=res_all )
food_rating = overall_rating.copy()
service_rating = overall_rating.copy() 

for r, u, o, f, s in zip(rating.placeID, rating.userID, rating.rating, rating.food_rating, 
                         rating.service_rating):
    overall_rating.loc[r,u] = o
    food_rating.loc[r,u] = f
    service_rating.loc[r,u] = s

# This tells us whether a restaurant-user pair has a rating. 0 means No and 1 means Yes.
review = pd.DataFrame( np.zeros(overall_rating.shape), columns=user_all, index=res_all)
review[overall_rating >= 0] = 1

# use dummy variables for different cuisine categories of the restaurants
res_cuisine = pd.get_dummies(cuisine,columns=['Rcuisine'])

# remove duplicate restaurant ID's. 
# A restaurant with multiple cuisine categories would have multiple columns equal 1
res_cuisine = res_cuisine.groupby('placeID',as_index=False).sum()

# use dummy variables for different cuisine categories of the restaurants
res_parking = pd.get_dummies(parking,columns=['parking_lot'])

# remove duplicate restaurant ID's. 
# A restaurant with multiple parking options would have multiple columns equal 1
res_parking = res_parking.groupby('placeID',as_index=False).sum()

geo.columns.values

# These are the ones that I think might be relevant
res_features = geo[['placeID','alcohol','smoking_area','other_services','price','dress_code',
               'accessibility','area']]

df_res = pd.DataFrame({'placeID': res_all})
df_res = pd.merge(left=df_res, right=res_cuisine, how="left", on="placeID")
df_res = pd.merge(left=df_res, right=res_parking, how="left", on="placeID")
df_res = pd.merge(left=df_res, right=res_features, how="left", on="placeID")

# The placeID's for the 130 restaurants with ratings
res_rated = res_all[np.sum(review,axis=1) > 0] 

# tells us whether a restaurant-user pair has a rating. 0 means No and 1 means Yes.
R = review.loc[res_rated].values  # shape = (130,138)

# These also have a shape of (130, 138)
Y_overall = overall_rating.loc[res_rated].values
Y_food  = food_rating.loc[res_rated].values
Y_service = service_rating.loc[res_rated].values

# select the indices of "df_res" where a restaurant has ratings
index = [x in res_rated for x in df_res['placeID'].values]

# restaurant features for the 130 restaurants with ratings
X = df_res.loc[index, :].reset_index(drop=True)

X.isnull().sum() # all the NANs are from cuisine 

# fill all NANs with 0
X = X.fillna(0) 

# drop a feature if the entire column are 0
features_to_drop = X.columns.values[np.sum(X,axis=0) == 0] 
X = X.drop(features_to_drop, axis=1)

# drop placeID
X = X.drop(['placeID'], axis=1)

# There are the restaurant features we'll explore
X.columns.values

n_res = R.shape[0]
n_user = R.shape[1]

## parking; the following values were simply chosen by some trail and error 
X['parking'] = np.zeros(n_res)

index = X.parking_lot_none == 1
X.loc[index,'parking'] = 0

index = X.parking_lot_yes == 1
X.loc[index,'parking'] = 2

index = X['parking_lot_valet parking'] == 1
X.loc[index,'parking'] = 2

index = X.parking_lot_public == 1
X.loc[index,'parking'] = 1

X = X.drop(['parking_lot_none','parking_lot_valet parking','parking_lot_yes','parking_lot_public'], axis=1)

## alcohol
X = pd.get_dummies(X,columns=['alcohol'])
# drop one variable to avoid the "dummy variable trap"
X = X.drop(['alcohol_No_Alcohol_Served'], axis=1) 

## price
X = pd.get_dummies(X,columns=['price'])
# drop one variable to avoid the "dummy variable trap"
X = X.drop(['price_low'], axis=1)

## smoking_area; the following values were simply chosen by some trail and error
X.smoking_area = X.smoking_area.map({'none':0, 'not permitted':-1, 'section': 1, 'permitted':1, 'only at bar':1})

## other services
# X = pd.get_dummies(X,columns=['other_services'])
# X = X.drop(['other_services_none'], axis=1)
X = X.drop(['other_services'], axis=1)

## accessibility
# X = pd.get_dummies(X,columns=['accessibility'])
# X = X.drop(['accessibility_no_accessibility'], axis=1) 
X = X.drop(['accessibility'], axis=1)

## dress_code
# X.dress_code = X.dress_code.map({'informal':0, 'casual':0, 'formal': 1})
X = X.drop(['dress_code'], axis=1)

## area
# X.area = X.area.map({'closed':0, 'open':1})
X = X.drop(['area'], axis=1)

## drop Rcuisine
X = X[X.columns[23:]]

# Add a bias term
X['x0'] = 1 # bias term

# 30% of the existing ratings will be used as the test set, so during 
# the training, they will be flagged. 
#
# The minimum number of ratings from a user = 3. In such a case, it 
# will be a 2/1 split.

random.seed(99)
cond = True

while cond:

    R_train = R.copy()

    # loop over each user
    for i in range(R_train.shape[1]):
        # the restaurants that are rated
        index = list( np.where(R_train[:,i] == 1)[0] )  
        # randomly select about 30% of them to be flagged
        flag = int(round(len(index)*0.3))
        index_flag = random.sample(index,flag)
        R_train[index_flag,i] = 0  
    
    # make sure in the traning set, each restaurant also has at least 
    # 2 ratings
    if np.sum(R_train,axis=1).min() > 1: 
        cond = False

# the rest will be the test set        
R_test = R - R_train

# Now "R_train" contains 810 ones and "R_test" contains 351 ones ("R" contains 1161 ones)
print(R_train.sum(), R_test.sum())

def RMSE(Y,Y_pred,R):
    
    return np.sqrt(mean_squared_error(Y[R  == 1], Y_pred[R == 1]))


def FCP(Y,Y_pred,R,verbose=True):
    
    # list of true ratings from each user (we only select users with at least two ratings)
    Y_fcp = []  
    Y_pred_fcp = [] # list of predicted ratings from each user 
    n_user = R.shape[1]
    
    for i in range(n_user):
        
        cond = (R.sum(axis=0) >= 2)[i] # there should be at least two ratings from a user
        index = np.where( R[:,i] == 1)[0] # the indices (restaurants) with ratings
    
        if cond:
            
            Y_fcp.append( (Y*R)[:,i][index] )
            Y_pred_fcp.append( (Y_pred*R)[:,i][index] )

        
    n_fcp = len(Y_fcp) # number of users with at least two ratings
    TP = 0. # Total number of pairs
    DP = 0. # number of discordant pairs
    CP = 0. # number of concordant pairs (excluding ties)
    
    for i in range(n_fcp):
        
        num_Y = len(Y_fcp[i])   # number of ratings from a user
        TP += num_Y*(num_Y-1)/2 # number of rating pairs = n*(n+1)/2 

        greater = np.array([])
        greater_pred = np.array([])

        # this loop is to go over all the rating pairs
        for j in range(num_Y-1):
            
            not_equal = Y_fcp[i][j] != Y_fcp[i][j+1:]
            greater = Y_fcp[i][j] > Y_fcp[i][j+1:]
            greater_pred = Y_pred_fcp[i][j] > Y_pred_fcp[i][j+1:]

            # filter the ones that are not ties
            greater = greater[not_equal]
            greater_pred = greater_pred[not_equal]

            DP += (greater != greater_pred).sum()
            CP += (greater == greater_pred).sum()

    if verbose:        
        print("Total number of rating pairs: {}".format(int(TP)))
        print("Total number of discordant pairs: {}".format(int(DP)))
        print("Total number of concordant pairs: {}".format(int(CP)))
        print("Total number of ties: {}".format(int(TP-DP-CP)))
        print("FCP: {}".format(CP/(CP+DP)))
    
    return CP/(CP+DP)

"""
The parameters of the cost function are the weights of the users, with a 
shape = (n_user, n_feature).However, to feed the cost function to SciPy's 
minimize(), the parameters of the function cannot be a matrix and has to 
be a 1D vector
"""

def CostFunction(params, X, Y, R, lambd):

    num_user = R.shape[1] # number of users
    num_feature = X.shape[1] # number of features
    
    # reshape the parameters to a 2D matrix so we can perform matrix multiplication
    Theta = params.reshape(num_user, num_feature)
    J = 0.5 * np.sum( (np.dot(X, Theta.T) * R - Y)**2 )

    # regularization; the bias term is not included 
    J = J + lambd/2. * np.sum(Theta[:,:-1]**2) 

    return J


def Gradient(params, X, Y, R, lambd):

    num_user = R.shape[1]
    num_feature = X.shape[1]

    Theta = params.reshape(num_user, num_feature)
    Theta_grad = np.dot((np.dot(Theta, X.T) * R.T - Y.T), X)

    # regularization
    Theta_grad[:,:-1] = Theta_grad[:,:-1] + lambd*Theta[:,:-1]

    return Theta_grad.reshape(-1)

def MeanNorm(Y,R):
    
    Y_norm = Y*R
    mean =  (np.sum(Y_norm, axis=1)/np.sum((R == 1.0), axis=1)).reshape(Y.shape[0],1) * np.ones(Y.shape)
    Y_norm = (Y_norm - mean)*R

    return Y_norm, mean

# This value is what I found that lead to the best FCP
lambd = 16.

# initialize the parameters and unroll them to a 1-D vector
np.random.seed(0)
n_feature = X.shape[1]
Theta = np.random.normal(0,1,(n_user, n_feature)).reshape(-1)

# Let's try to predict the overall ratings
Y = Y_overall 

# mean normalization
Y_norm, Y_mean = MeanNorm(Y,R_train)

# optimization
result = minimize(CostFunction, Theta, jac=Gradient, args=(X, Y_norm, R_train, lambd),
                  options={'disp': True, 'maxiter': 500})

# reshape the optimial parameters to a 2D matrix
Theta_opt = result.x.reshape(n_user, n_feature)
Y_pred = np.dot(X, Theta_opt.T) + Y_mean

# A plotter to make boxplot
def MakeBoxplot(Y_pred, Y_true, R, title):
    
    data1 = Y_pred[R == 1][Y_true[R == 1] == 0]
    data2 = Y_pred[R == 1][Y_true[R == 1] == 1]
    data3 = Y_pred[R == 1][Y_true[R == 1] == 2]
    data = [data1,data2,data3]

    fig = plt.figure()
    plt.boxplot(data)
    plt.xticks([1, 2, 3],[0,1,2])
    plt.xlabel('True Rating')
    plt.ylabel('Predicted Rating')
    plt.title(title)
    plt.show()

# RMSE
print("RMSE of the training set: {}".format(RMSE(Y,Y_pred,R_train)))
print("RMSE of the test set: {}".format(RMSE(Y,Y_pred,R_test)))

# FCP
print("Training Set:")
FCP(Y,Y_pred,R_train)
print("\n")
print("Test Set:")
FCP(Y,Y_pred,R_test)

MakeBoxplot(Y_pred, Y, R_train, 'Training Set')

MakeBoxplot(Y_pred, Y, R_test, 'Test Set')



