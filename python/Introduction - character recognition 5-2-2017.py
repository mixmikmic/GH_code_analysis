from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

digits = datasets.load_digits()

image0 = digits.images[0]    # a single digit image
#images are ndarray
image0.shape

images_and_labels = list(zip(digits.images, digits.target))  # target is the classification label
len(images_and_labels)  # how many images

images_and_labels[0]

for index, (image, label) in enumerate(images_and_labels[:4]):    # plot first 4 images and their label
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

n_samples = len(digits.images)
n_samples

data = digits.images.reshape((n_samples, -1))      # flatten the image using np.reshape
data.shape

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

test_data = data[:round(n_samples / 2)]
test_data_labels = digits.target[:round(n_samples / 2)]

# We learn the digits on the first half of the digits
classifier.fit(test_data, test_data_labels)

# Now predict the value of the digit on the second half:
expected = digits.target[round(n_samples / 2):]                # labels of the second half of the data
predicted = classifier.predict(data[round(n_samples / 2):])    # run the classifier

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[round(n_samples / 2):], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()

from sklearn.externals import joblib
joblib.dump(classifier, 'classifier.pkl')     # save the classifier as a pickle object

ls



