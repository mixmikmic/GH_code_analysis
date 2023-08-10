import numpy as np
np.random.seed(0)

# We'll use this method rather than use the Pythagorean theorem to calculate Euclidean distance manually.  
from scipy.spatial import distance

class KNN():
    
    def _euc(self, a, b):
        '''Helper function that returns the Euclidean distance individual points a and b '''
        return distance.euclidean(a, b)
    
    def _distance(self, x1, x2):
        """ Calculates the l2 distance between two vectors of the same length.  Loops through each 
        corresponding row in the vectors and uses the _euc() helper method at each iteration to get the sum of 
        the distance between all corresponding rows in the two vectors.  
        (x1[0] -> x2[0], x1[1] -> x2[1]...x1[n] -> x2[n])"""
        distance = 0
        for i in range(len(x1)):
            distance += self._euc(x1[i], x2[i])
        return distance
    
    def fit(self, X_train, y_train):
        """Stores values for X_train and y_train."""
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        """Predicts the label of the class for each row in X_test, based on the single closest point to each. 
        The way this is currently written, K is hardcoded to one so the user does not pass in a value for K when 
        calling it.
        
        TODO:  Modify this function and the function below so that the user can pass in a value for K, so that 
        the algorithm makes it's prediction based on the K nearest neighbors (rather than the single closest 
        neighbor as it is currently written.)  You will need to modify this function AND _closest() to 
        complete this task."""
        predicted_classes = []
        for point in X_test:
            predicted_classes.append(self._closest_label(point))
        return predicted_classes
        
    def _closest(self, row):
        """Modify this function so that it takes a parameter, K, and retrieves the K closest points.  As is, this
        function currently only retrieves the labels of the single closest point.  """
        
        shortest_distance = self._distance(row, self.X_train[0])
        for i in range(1, len(self.X_train)):
            current_distance = self._distance(row, self.X_train[i])
            index_of_closest = 0
            # If this distance is shorter than the shortest distance, make it the new shortest distance.  
            if current_distance < shortest_distance:
                shortest_distance < current_distance
                index_of_closest = i
        return y_train[index_of_closest]
    

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# use load_iris() to bind the data to the 'iris' variable. The iris object will contain the data in it's .data 
#attribute, and the labels for the data in it's .target attribute
iris = None
X_vals = None
y_vals = None

# Create a StandardScaler() Object. 
scaler = None

# Call scaler.fit() on the X_vals that will be rescaled.

# Bind the newly scaled X_vals to scaled_X_vals by calling scaler.transform() on X_vals.
scaled_X_vals = None

from sklearn.model_selection import train_test_split

# TODO: Use train_test_split to split our scaled data into training and testing sets.  Use a test amount of .5.  

#TODO: Create a KNN object using the class you wrote above.  Fit the data and use it to predict labels for X_test.  
clf = None



from sklearn.metrics import accuracy_score

# TODO: Use the accuracy_score object to evaluate the quality of your KNN object's predictions. Try passing in different
# values for K at prediction time and see which value does best!



# TODO: Finally, get some practice with SKLearn's KNN Classifier.  Import the KNeighborsClassifier object from 
# sklearn.neighbors.  Fit the data to the object and use it to make predictions just as you did with your own KNN above. 
# Finally, use the accuracy_score object to measure the quality of this classifier's predictions. 



