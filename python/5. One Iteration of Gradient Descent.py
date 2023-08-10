# One iteration of Gradient Descent. Page 58
# Step 1: An Empty Network

weight=0.1
alpha=0.01

def nerual_network(inputs,weight):
    prediction=inputs*weight
    return prediction

# Step 2: Making A Prediction And Evaluating Error
number_of_toes=[8.5]
win_or_lose_binary=[1]

inputs=number_of_toes[0]
goal_pred=win_or_lose_binary[0]

pred=nerual_network(inputs,weight)
error=(pred-goal_pred)**2
print error

# Step 3:Compare: calculate 'node delta' and put it on the output node
delta=pred-goal_pred
print delta

# Step 4:Learn: Calculating 'weight delta' and putting it on the weight

weight_delta=inputs*delta
print weight_delta

# Step 5: Learn,Updating the weight
weight-=weight_delta*alpha
print weight



