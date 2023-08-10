import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
get_ipython().magic('matplotlib inline')

import glob

rect_tag_files = glob.glob('data/I/*.png')
circle_tag_files = glob.glob('data/O/*.png')
queen_tag_files = glob.glob('data/Q/*.png')

rect_image =    cv2.imread(rect_tag_files[0], cv2.IMREAD_GRAYSCALE)
circle_image =  cv2.imread(circle_tag_files[0], cv2.IMREAD_GRAYSCALE)
queen_image =   cv2.imread(queen_tag_files[0], cv2.IMREAD_GRAYSCALE)

plt.figure(figsize = (10, 7))
plt.title('Rectangle Tag')
plt.axis('off')
plt.imshow(rect_image,  cmap = cm.Greys_r)

plt.figure(figsize = (10, 7))
plt.title('Circle Tag')
plt.axis('off')
plt.imshow(circle_image,  cmap = cm.Greys_r)

plt.figure(figsize = (10, 7))
plt.title('Queen Tag')
plt.axis('off')
plt.imshow(queen_image,  cmap = cm.Greys_r)

rect_tag_class = len(rect_tag_files) * [1]
circle_tag_class = len(circle_tag_files) * [2]
queen_tag_class = len(queen_tag_files) * [3]

print(len(rect_tag_files), len(rect_tag_class), rect_tag_files[0], rect_tag_class[0])
print(len(circle_tag_files), len(circle_tag_class), circle_tag_files[0], circle_tag_class[0])
print(len(queen_tag_files), len(queen_tag_class), queen_tag_files[0], queen_tag_class[0])

all_tag_files = []
all_tag_files.extend(rect_tag_files)
all_tag_files.extend(circle_tag_files)
all_tag_files.extend(queen_tag_files)

all_classifications = []
all_classifications.extend(rect_tag_class)
all_classifications.extend(circle_tag_class)
all_classifications.extend(queen_tag_class)

all_images = []
for image_file in all_tag_files:
    read_image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    all_images.append(read_image)

print("Do the number of images and the number of classifications in the two lists match up?")
print(len(all_images), len(all_classifications))

test_images = [rect_image, circle_image, queen_image]

def modify_image(img):
    #img = cv2.blur(image, (5, 5))
    #img = cv2.GaussianBlur(image, (5, 5), 0)
    img = cv2.medianBlur(image, 5)
    #img += 10
    img * 1.9
    
    img = img[4:20,4:20]
    return img

for image in test_images:
    image = modify_image(image)
    plt.figure(figsize = (15, 12))
    plt.axis('off')
    plt.imshow(image, cmap = cm.Greys_r)

all_images_flat = []
for image in all_images:
    mod_image = modify_image(image)
    flat_image = mod_image.flatten()
    all_images_flat.append(flat_image)

X = np.array(all_images_flat)
y = np.array(all_classifications)

print(X.shape, y.shape)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=4)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
fit_trans_X = pca.fit(X).transform(X)
plt.figure(figsize = (35, 20))
plt.scatter(fit_trans_X[:, 0], fit_trans_X[:, 1], c=y, s=400)

from sklearn.lda import LDA

lda = LDA(n_components=2)
lda_model = lda.fit(X_train, y_train)
X_trans = lda_model.transform(X_train)
plt.figure(figsize = (35, 20))
plt.scatter(X_trans[:, 0], X_trans[:, 1], c=y_train, s=400)

from sklearn import svm

clf = svm.SVC(gamma=0.0001, C=10)
clf.fit(X_trans, y_train)

transform_testing_set = lda.transform(X_test)
y_pred = clf.predict(transform_testing_set)

from sklearn import metrics

print (metrics.accuracy_score(y_test, y_pred))



