weight=0.5
goal_pred=0.8
inputs=2
alpha=0.1

for iteration in range(20):
    prediction=inputs*weight
    error=(prediction-goal_pred)**2
    derivative=(prediction-goal_pred)*inputs
    weight-=alpha*derivative
    
    print 'Iteration:'+str(iteration)+'    Erroe: '+str(error)+'   Prediction: '+str(prediction)

weight=0.5
goal_pred=0.8
inputs=2
alpha=0.01

for iteration in range(20):
    prediction=inputs*weight
    error=(prediction-goal_pred)**2
    derivative=(prediction-goal_pred)*inputs
    weight-=alpha*derivative
    
    print 'Iteration:'+str(iteration)+'    Erroe: '+str(error)+'   Prediction: '+str(prediction)



