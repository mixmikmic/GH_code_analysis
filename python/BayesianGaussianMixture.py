import numpy as np
import csv
from sklearn.mixture import BayesianGaussianMixture

def bgm(path_and_name, ratio):
    
    # Turn csv into numpy array and remove the header with column names.
    original = np.genfromtxt(path_and_name, delimiter=',')
    headless = original[1:]

    # Select on which features to cluster on:
    # 0 = DBZH
    # 1 = TH
    # 2 = VRAD
    # 3 = X coordinate
    # 4 = Y coordinate
    # 5 = Z coordinate
    data = headless[:,(0,2,3,4,5)]

    # Randomly shuffle the data and divide it up in a training and test set.
    length = np.shape(data)[0]
    div = int(length * ratio)
    np.random.shuffle(data)

    trainset = data[:div]
    testset = data[div:]

    # Specify mixture settings and fit on trainingset
    # Algorithm works best on current data with either, more components and a lower prior or,
    # less components and a higher prior.
    # Least n_components needs to be 2.
    mix = BayesianGaussianMixture(n_components=6,max_iter=10000,weight_concentration_prior_type='dirichlet_distribution', weight_concentration_prior=0.00001).fit(trainset)
    print("Trainingsset fitted")

    # Predict testset
    labels = mix.predict(testset)
    print("Testset predicted")

    # Append labels to data and write to file.
    labeledData = np.append(testset, np.reshape(labels, (-1,1)), axis = 1)
    
    name = path_and_name.split("/")
    name = name[len(name)-1]
    name = "BGM_" + name
    np.savetxt("output/"+name, labeledData, delimiter=',')

bgm('csvdata/160930/160930-23-00.csv', 0.4)



