import numpy as np

def classify_rain(path_and_name):
    # Generate array from csv and extract DBZH vector.
    original = np.genfromtxt(path_and_name, delimiter=',')
    data = original[1:]
    
    # Initiate list to keep track of classes
    classification = np.array([])

    # Loop over DBZH value, add a 0 to the list if it is rain (DBZH > 7.0),
    # else add a 1.
    for i in data[:,0]:
        if(i > 7.0):
            classification = np.append(classification, 0)
        else:
            classification = np.append(classification, 1)

    # Reshape into an appendable column
    classification = np.reshape(classification, (-1,1))


    # Add labels to data, output format: DBZH, X, Y, Z, labels
    output = np.append(data[:,(0,2,3,4,5)], classification, axis=1)

    # Write to CSV
    name = path_and_name.split("/")
    name = name[len(name)-1]
    name = "Naive_" + name
    np.savetxt("output/"+name, output, delimiter=',')

classify_rain("csvdata/161007/161007-20-30.csv")



