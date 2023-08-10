# Use the numpy library.
import numpy as np

# Let's set the labels as: positive = 1, and negative = 0
# So if we have 3 labels set to [0,1,0], this indicates, [negative_class, positive_class, negative_class]
# This assumes are are working with a binary classification problem!

# These are the labels we predicted.
pred_labels = np.asarray([0,1,1,0,1,0,0])
print 'pred labels:\t\t', pred_labels

# These are the true labels.
true_labels = np.asarray([0,0,1,0,0,1,0])
print 'true labels:\t\t', true_labels

# True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))

# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))

# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

print 'TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN)

from sklearn.metrics import confusion_matrix

print confusion_matrix(true_labels, pred_labels)
print '[[TN,FP]'
print '[[FN,TP]]'

# Let's start with True Positives.
# TP = we predicted a label of 1, and the true label is 1.

# So let's get all the cases where the predicted labels are positive (i.e., 1)
pred_labels_pos = pred_labels == 1
# Convert from True/False to 1/0
pred_labels_pos = pred_labels_pos.astype(np.int)
print 'pred_labels_pos:\t', pred_labels_pos 

# Now let's get all the cases where the true labels are also 1.
true_labels_pos = true_labels == 1
true_labels_pos = true_labels_pos.astype(np.int)
print 'true_labels_pos:\t', true_labels_pos
print '    AND operation\t-------'

# Now we get the cases where the pred_labels and true_labels are both positive (indicated with a '1')
# To do so, we can use the logical AND operation.
pred_pos_AND_true_pos = np.logical_and(pred_labels_pos, true_labels_pos)
pred_pos_AND_true_pos = pred_pos_AND_true_pos.astype(np.int)
print 'pred_pos_AND_true_pos:\t', pred_pos_AND_true_pos


# We now have indicated all the true positives with a 1. 
# To compute the number of true positives, we can just sum over the array.
TP = np.sum(pred_pos_AND_true_pos)
print 'TP: ', TP 

# Now let's look at how to compute False positives (FP).
# FP = we predicted a label of 1, but the true label is 0.

# We already figured out all the cases where we predicted a positive label.
print 'pred_labels_pos:\t', pred_labels_pos 

# Get all the cases where the true labels are negative (i.e.,0)
true_labels_neg = true_labels == 0
true_labels_neg = true_labels_neg.astype(np.int)
print 'true_labels_neg:\t', true_labels_neg

print '    AND operation\t-------'

# To get cases where the pred_labels are positive and the true_labels are negative.
# Again we use the logical AND operation.
pred_pos_AND_true_neg = np.logical_and(pred_labels_pos, true_labels_neg)
pred_pos_AND_true_neg = pred_pos_AND_true_neg.astype(np.int)
print 'pred_pos_AND_true_neg:\t', pred_pos_AND_true_neg

# We now have indicated all the false positives with a 1. 
# To compute the number of false positives, we can just sum over the array.
FP = np.sum(pred_pos_AND_true_neg)
print 'FP: ', FP 

# Okay, now what about True Negatives? 
# TN = we predicted a label of 0, and the true label is 0.

pred_labels_neg = pred_labels == 0
pred_labels_neg = pred_labels_neg.astype(np.int)
print 'pred_labels_neg:\t', pred_labels_neg

# We already computed when the true labels are negative (0).
print 'true_labels_neg:\t', true_labels_neg

# Again we use the logical AND operation.
print '    AND operation\t-------'
pred_neg_AND_true_neg = np.logical_and(pred_labels_neg, true_labels_neg)
pred_neg_AND_true_neg = pred_neg_AND_true_neg.astype(np.int)
print 'pred_neg_AND_true_neg:\t', pred_neg_AND_true_neg

# Again, sum to count how many true negatives we have.
TN = np.sum(pred_neg_AND_true_neg)
print 'TN: ', TN

# Finally, let's compute False Negatives.
# FN = we predict 0, but the true label is 1.

# Already computed when the predicted labels are negative.
print 'pred_labels_neg:\t', pred_labels_neg

# Already computed when the true labels are positive (1).
print 'true_labels_pos:\t', true_labels_pos

# Use logical AND.
print '    AND operation\t-------'
pred_neg_AND_true_pos = np.logical_and(pred_labels_neg, true_labels_pos)
pred_neg_AND_true_pos = pred_neg_AND_true_pos.astype(np.int)
print 'pred_neg_AND_true_pos:\t', pred_neg_AND_true_pos

# Again, sum to count how many true negatives we have.
FN = np.sum(pred_neg_AND_true_pos)
print 'FN: ', FN

