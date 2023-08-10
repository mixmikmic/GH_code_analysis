import numpy as np

# Data information
streetlights=np.array([[1,0,1],
                [0,1,1],
                [0,0,1],
                [1,1,1],
                [0,1,1],
                [1,0,1]])

walk_vs_stop=np.array([[0],
                     [1],
                     [0],
                     [1],
                     [1],
                     [0]])
weights=np.array([0.5,0.48,-0.7])

alpha=0.1

inputs=streetlights[0]
goal_prediction=walk_vs_stop[0]

# Train the model

for iteration in range(40):
    error_for_all_lights=0
    for index in range(len(walk_vs_stop)):
        inputs=streetlights[index]
        goal_prediction=walk_vs_stop[index]
        
        prediction=inputs.dot(weights)
        error=(goal_prediction-prediction)**2
        error_for_all_lights+=error
        
        delta=prediction-goal_prediction
        weights-=alpha*(inputs*delta)
        
        print 'Prediction: '+str(prediction)
        print 'Error: '+str(error_for_all_lights)+'\n'
        



