import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

get_ipython().magic("run 'Data_Munging.ipynb'")

def plot_confusion_matrix(cm, kfold, l, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(str(kfold) + '-fold cross validation - ' + title)
    plt.colorbar()
    tick_marks = np.arange(len(l))
    plt.xticks(tick_marks,activityLabel.values, rotation=45)
    plt.yticks(tick_marks, activityLabel.values)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def calculateErrorRate(predictLabel, labelTestingData):
    error_rate = 0.0
    for i in xrange(len(predictLabel)):
        if predictLabel[i] != labelTestingData[i]: 
            error_rate += 1
    return error_rate/len(predictLabel)

def crossValidation(kfold):
    dayList = [sensorData['Start time'][x].day for x in xrange(1,len(sensorData))]
    dayList= list(set(dayList))
    l = list(activityLabel.index)
    errorRate = 0
    confusionMatrix = confusion_matrix(activityLabel, activityLabel, labels = l)
    confusionMatrix = confusionMatrix - confusionMatrix #we have then a 0 confusion matrix
    
    for fold in xrange(kfold): 
        testingDay = 26
        while testingDay == 26:
            randomNumber = randint(0,len(dayList)-1)
            testingDay = dayList[randomNumber]
        print testingDay
        
        trainingSensor = [sensorData['Start time'][x].day != testingDay and sensorData['End time'][x].day != testingDay for x in range(1,len(sensorData))]
        trainingSensor = trainingSensor + [False]
        trainingSensorData = sensorData[trainingSensor]
        trainingSensorData.index = np.arange(1,len(trainingSensorData)+1)
            
        trainingActivity = [activityData['Start time'][x].day != testingDay and activityData['End time'][x].day != testingDay for x in range(1,len(activityData))]
        trainingActivity = trainingActivity + [False]
        trainingActivityData = activityData[trainingActivity]
        trainingActivityData.index = np.arange(1,len(trainingActivityData)+1)
        
        trainingFeatureMatrix, trainingLabel = convert2LastFiredFeatureMatrix(trainingSensorData,trainingActivityData, 60)
        cumuSensor, cumuActivity = cumulationTable(trainingFeatureMatrix, trainingLabel) 
        sensorTrainingData = np.asarray(cumuSensor)
        labelTrainingData = np.asarray(cumuActivity)
        
        daySensor = [sensorData['Start time'][x].day == testingDay and sensorData['End time'][x].day == testingDay for x in range(1,len(sensorData))]
        daySensor = daySensor + [False]
        daySensorData = sensorData[daySensor]
        daySensorData.index = np.arange(1,len(daySensorData)+1)

        dayActivity = [activityData['Start time'][x].day == testingDay and activityData['End time'][x].day == testingDay for x in range(1,len(activityData))]
        dayActivity = dayActivity + [False]
        dayActivityData = activityData[dayActivity]
        dayActivityData.index = np.arange(1,len(dayActivityData)+1)
        
        testingFeatureMatrix, testingLabel = convert2LastFiredFeatureMatrix(daySensorData,dayActivityData,60)
        cumuSen, cumuAct = cumulationTable(testingFeatureMatrix, testingLabel)
        sensorTestingData = np.asarray(cumuSen)
        labelTestingData = np.asarray(cumuAct)
        
        knn = KNeighborsClassifier(n_neighbors=4,p=2, metric='minkowski')
        knn.fit(sensorTrainingData, labelTrainingData)
        predictLabel = knn.predict(sensorTestingData)
        
        errorRate += calculateErrorRate(predictLabel, labelTestingData)        
        confusionMatrix += confusion_matrix(labelTestingData, predictLabel, labels = l)
        print 'Turn {0}, error rate: {1}'.format(fold, calculateErrorRate(predictLabel, labelTestingData))
        print 'Confusion matrix:'
        print confusion_matrix(labelTestingData, predictLabel, labels = l)
    
    
    print 'Error rate:',float(errorRate)/kfold
    
    np.set_printoptions(precision=2)
    print('{0}-fold cross validation - Confusion matrix, without normalization').format(kfold)
    print(confusionMatrix)
    plt.figure()
    plot_confusion_matrix(confusionMatrix, kfold,l)

    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = confusionMatrix.astype('float')/confusionMatrix.sum(axis=1)[:, np.newaxis]
    print('{0}-fold cross validation - Normalized confusion matrix').format(kfold)
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, kfold,l, title='Normalized confusion matrix')

    plt.show()
        
        

crossValidation(10)













