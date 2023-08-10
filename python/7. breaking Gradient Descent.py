weight=0.5
goal_pred=0.8
inputs=2

# Train the model

for iteration in range(20):
    prediction=inputs*weight
    error=(prediction-goal_pred)**2
    weight_delta=inputs*(prediction-goal_pred)
    weight-=weight_delta
    
    print 'Error: '+str(error)+'      Prediction: '+str(prediction)
    

