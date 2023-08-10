from IPython.display import Image, display
display(Image(filename='SelfOrgMap.png', embed=True))

display(Image(filename='SOMIntuition.png', embed=True))

display(Image(filename='RGBreduce.png', embed=True))

display(Image(filename='WorldPoverty.png', embed=True))

# Import dependencies
# numpy for matrix algbera
import numpy as np
# Pandas for data manipulation
import pandas as pd
# matplotlib for data visualization
import matplotlib.pyplot as plt

# Set seed so we get same random allocation on each run of code
np.random.seed(2017)

# Load the IMDB data
educ_data = pd.read_csv("Grade1Students.csv")

# Show structure of data set
educ_data.head()

# Summary statistics for data set (mean, stdev, min, max, etc)
educ_data.describe()

# We will normalize each feature to have mean 0 and standard deviation 1
# This standardization is done to represent input data on the same scale

# Standardize free lunch status
educ_data["g1freelunch"] = (educ_data["g1freelunch"]-np.mean(educ_data["g1freelunch"]))/np.std(educ_data["g1freelunch"])

# Standardize absences 
educ_data["g1absent"] = (educ_data["g1absent"]-np.mean(educ_data["g1absent"]))/np.std(educ_data["g1absent"])

# Standardize reading score
educ_data["g1readscore"] = (educ_data["g1readscore"]-np.mean(educ_data["g1readscore"]))/np.std(educ_data["g1readscore"])

# Standardize math score
educ_data["g1mathscore"] = (educ_data["g1mathscore"]-np.mean(educ_data["g1mathscore"]))/np.std(educ_data["g1mathscore"])

# Standardize listening score
educ_data["g1listeningscore"] = (educ_data["g1listeningscore"]-np.mean(educ_data["g1listeningscore"]))/np.std(educ_data["g1listeningscore"])

# Standardized word study score
educ_data["g1wordscore"] = (educ_data["g1wordscore"]-np.mean(educ_data["g1wordscore"]))/np.std(educ_data["g1wordscore"])

# Initialize total number of itterations (remember n = 5550)
total_itter = 3*len(educ_data.index)

# Initialize number of output nodes
nodes_num = 3

# Dimension of input data
input_dim = len(educ_data.columns)

# Initialize parameters for learning rate 
learn_init = 0.1

# Step 1: Initialize the weight vectors 
# Randomly generated matrix with entries between [-2,2], each column is a weight vector 
Weight_mat = 4*np.random.rand(input_dim,nodes_num)-2

# Show initialized weight matrix
print "Initialized weight matrix,", Weight_mat

# Start SOM algorithm itterations
for itter in range(total_itter):
    
    # Initialize distance from weight to chosen point (will be updated in inner loop)
    dist_bmu = float("inf")
    
    # Step 2: Choose data point at random from input data
    
    # Select row index at random
    row_index = np.random.randint(len(educ_data.index))
    
    # Get corresponding data vector
    data_chosen = educ_data.loc[[row_index]]
    
    # Step 3: Find the weight vector that is closest to chosen point
    for node in range(nodes_num):
        
        # Compute euclidean distance from weight vector to chosen point
        dist_neuron = np.linalg.norm(data_chosen-Weight_mat[:,node])
        
        # Save the node with shortest distance of its neuron to chose point
        if dist_neuron < dist_bmu:
            
            # Update distance from weight to chosen point
            dist_bmu = dist_neuron
            
            # Best matching unit (BMU)
            weight_bmu = Weight_mat[:,node]
            index_bmu = node
            
    # Step 4: Define radius of winning neuron neighbourhood 
    # We skip this step because we only have 3 neurons in our application
    
    # Define learning rate
    learn_rate = learn_init*np.exp(-itter/total_itter)
    
    # Step 5: Update weight vectors (w_{t+1} = w_{t} + L(t)*(x_{i} - w_{t}))
    Weight_mat[:,index_bmu] = np.add(weight_bmu,learn_rate*(np.subtract(data_chosen,weight_bmu)))

# Show trained weights
print "Trained weights from SOM,", Weight_mat

# Initialize vector the classifies each student into group 1,2,3
group = np.zeros(len(educ_data.index))
    
# Classify input data
for index, data in educ_data.iterrows():
    
    # Initialize distance from cluster centroid
    dist_cluster = float("inf")
    
    # Find closest weight centroid
    for centroid in range(nodes_num):
        
        # Compute euclidean distance from centroid vector to data point
        dist_centroid = np.linalg.norm(data-Weight_mat[:,centroid])

        # Save centroid that is closest to data piont
        if dist_centroid < dist_cluster:

                # Update distance from weight to chosen point
                dist_cluster = dist_centroid

                # Best matching unit (BMU)
                group[index] = centroid+1
            
# Add group classifier column 
educ_data["group"] = group

# See labeled data (last column contains labels)
educ_data.head()

# Let us figure out which group is weak, average, strong

# For group 1:
# Notice the test scores are close to 0 standard deviations away from the mean
# This is likely to be the "average" group
educ_data[educ_data.group == 1].describe()

# For group 2:
# Notice that of students recieving free or reduced lunch much more than average, and
# are absent more than average, and have relatively lower test scores. 
#This is likely to be the "weak" group.
educ_data[educ_data.group == 2].describe()

# Four group 3:
# The student test scores much higher than on average in this group, students
# are from advantageous backgrounds as lower proportion of students on free
# or reduced price lunch. This is the "gifted" group.
educ_data[educ_data.group == 3].describe()

