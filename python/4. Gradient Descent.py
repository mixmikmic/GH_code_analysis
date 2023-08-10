weight=0.5
goal_pred=0.8
inputs=0.5

# Train the model
for iteration in range(50):
    pred=inputs*weight
    error=(pred-goal_pred)**2
    # Pure error: pred-goal_pred; inputs:scaling,negative reversal and stopping
    direction_and_amount=(pred-goal_pred)*inputs
    weight-=direction_and_amount
    
    print 'Interation'+str(iteration)+'    weight'+str(weight)+'    Error:'+str(error)+'    Prediction:'+str(pred)
    
    

