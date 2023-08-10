import numpy as np
import scipy
import scipy.stats
filename = 'raw_data_example.txt'

labels = []
labels2 = []
old = ""
with open(filename) as f:
    for line in f:
        line_acc = []
        if "B3BB" not in line:
            continue              
        if not int(line[0]) == 1:
            continue
        line = line.split(",")        
        if str(line[3]) == "Lying":
            continue      
        labels.append(line[5])
        l = line[5]
myset = set(labels)
#print "Labels: " + str(myset)
activities_set = list(myset)
dict_activity = dict(zip(activities_set, np.arange(len(activities_set))))  # {1:0, 2:1, 6:2, 32:3}
dict_activity

index2 = 0
index2+=1
print filename
data_timestamp = []
data_module = []
with open(filename) as f:
    i = 0
    for line in f:
        line_acc = []
        if "B3BB" not in line:
            continue
        if not int(line[0]) == 1:
            continue
        line = line.split(",")
        line_acc = [float(line[7]),float(line[8]),float(line[9]), int(dict_activity[line[5]])]
        line_acc = np.array(line_acc)
        line_acc = np.insert(line_acc, 0, i)
        line_acc.astype(float)            

        data_timestamp.append(line_acc)
        data_module.append(np.sqrt(line_acc[1]* line_acc[1] + line_acc[2]* line_acc[2] + line_acc[3]* line_acc[3]) )
        i+=1

data_timestamp = np.array(data_timestamp)

#EXTRACT THE WRIST DATA
columns_wrist = [0, 1, 2, 3]
data_Acc_lWrist = data_timestamp[:, columns_wrist]
data_Acc_lWrist = np.column_stack((data_timestamp[:, columns_wrist], np.array(data_module).T, data_timestamp[:, -1]))

labels = data_timestamp[:,-1]
myset = set(labels)

unique, counts = np.unique(labels, return_counts=True)
data_filtered_activities = data_Acc_lWrist[:]

#Segment the data... overlapping window
w_size = 100
overlap = 50
advancing = w_size - overlap

data_segmented = []
data_window = []
current_time = firts_time =data_filtered_activities[0][0]

for data_sample in data_filtered_activities:   
    if len(data_window) < w_size:
        data_window.append(data_sample.tolist())
    else:
        data_segmented.append(np.array(data_window))        
        data_window = data_window[advancing:]
        data_window.append(data_sample.tolist())        
#print len(data_segmented)

#Extract features
#[timestamp, feature1, feature2, ... , label]
data_features = []

for i, segment in enumerate(data_segmented):
    feature_vector = [i]
    for index, column in enumerate(segment.T):        
        if not (index == 0 or index == 1 or index == (len(segment.T)-1)):
            mean = np.mean(column)
            std = np.std(column)
            median = np.median(column)
            rms = np.sqrt(np.mean(np.square(column)))
            integral = np.trapz(column)
            #zero_crossing = ((column[:-1] * column[1:]) < 0).sum()
            energy = np.sum(column ** 2) / np.float64(len(column))
            kurtosis = scipy.stats.kurtosis(column)
            skewness = scipy.stats.skew(column)

            #feature_vector.extend([mean, std, energy])
            feature_vector.extend([mean, std, energy, median, rms, integral, kurtosis,skewness ])
    label = column[0]
    feature_vector.extend([label])
    data_features.append(feature_vector)

#np.savetxt('_data_JSI/data_features_null_overlap.csv', data_features, delimiter='\t', fmt='%5.6f') 
np.savetxt('data_features_example_'+ str(index2) +'.csv', data_features, delimiter=';', fmt='%5.6f') 
print len (data_features)



