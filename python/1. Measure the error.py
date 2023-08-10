# Measure the error
knob_weight=0.5
inputs=0.8
goal_pred=0.8

pred=inputs*knob_weight
error=(pred-goal_pred)**2

print error





