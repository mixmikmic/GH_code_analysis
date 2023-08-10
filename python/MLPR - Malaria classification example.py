from sklearn import model_selection, neighbors, metrics
get_ipython().run_line_magic('pylab', 'inline')

data = np.load('malaria-classification-example.npz')
X = data['X']
y = data['y']
images = data['images']

print('Loaded {0} images and labels.'.format(X.shape[0]))

imshow(images[0],cmap=cm.gray)

print('Class: {0}'.format(y[0]))
X[0]

pos = np.where(y==1)[0]
figsize(6,6)
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.imshow(images[pos[i],:,:], cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])

neg = np.where(y==0)[0]
figsize(6,6)
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.imshow(images[neg[i],:,:], cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

nn = neighbors.NearestNeighbors(n_neighbors=1)
nn.fit(X_train)

N_examples = 10
figsize(2,15)
for i in range(N_examples):
    subplot(N_examples,2,2*i+1)
    plt.imshow(np.reshape(X_test[i,:],(40,40)), cmap=plt.cm.gray)
    plt.title(y_test[i])
    plt.xticks([])
    plt.yticks([])
    
    subplot(N_examples,2,2*i+2)
    neighbour_idx = int(nn.kneighbors(X_test[i,:].reshape(1,-1))[1])
    plt.imshow(np.reshape(X_train[neighbour_idx,:],(40,40)), cmap=plt.cm.gray)
    plt.title(y_train[neighbour_idx])
    plt.xticks([])
    plt.yticks([])

clf = neighbors.KNeighborsClassifier(n_neighbors=1, weights='distance')
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:,1]
 
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

figsize(6,6)
plot(fpr, tpr)
title('ROC (area under curve=%.3f)' % (metrics.roc_auc_score(y_test, y_pred)))
xlabel('FPR')
ylabel('TPR')
grid(True)

score = []
max_k = 15
for k in range(1,max_k):
    clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance')
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:,1]
    score.append(metrics.roc_auc_score(y_test,y_pred))
    
plot(range(1,max_k),score)
grid(True)
plt.xlabel('k')
plt.ylabel('AUC')

mistakes = np.where(np.logical_xor(y_pred>.5,y_test))[0]

nn = neighbors.NearestNeighbors(n_neighbors=1)
nn.fit(X_train)

N_examples = 10
figsize(2,15)
for i in range(N_examples):
    subplot(N_examples,2,2*i+1)
    plt.imshow(np.reshape(X_test[mistakes[i],:],(40,40)), cmap=plt.cm.gray)
    plt.title(y_test[mistakes[i]])
    plt.xticks([])
    plt.yticks([])
    
    subplot(N_examples,2,2*i+2)
    neighbour_idx = int(nn.kneighbors(X_test[mistakes[i],:].reshape(1,-1))[1])
    plt.imshow(np.reshape(X_train[neighbour_idx,:],(40,40)), cmap=plt.cm.gray)
    plt.title(y_train[neighbour_idx])
    plt.xticks([])
    plt.yticks([])

score = []
train_set_size = np.arange(100,X_train.shape[0]+1,100)
k = 5
for N_train in train_set_size:
    clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform')
    clf.fit(X_train[:N_train,:], y_train[:N_train])
    y_pred = clf.predict_proba(X_test)[:,1]
    score.append(metrics.roc_auc_score(y_test,y_pred))
    
figsize(6,6)
plot(train_set_size,score)
grid(True)
xlabel('N_train')
ylabel('AUC')

images_aug = []
y_train_aug = []
X_train_aug = []

for i in range(y_train.shape[0]):
    for n_rotations in range(4):
        for n_flips in range(2):
            new_image = X_train[i].reshape(40,40)
            if n_flips==1:
                new_image = np.fliplr(new_image)
            for i_rotation in range(n_rotations):
                new_image = np.rot90(new_image)
            images_aug.append(new_image)
            y_train_aug.append(y_train[i])
            X_train_aug.append(new_image.ravel())
            
X_train = np.array(X_train)

figsize(10,6)
for i in range(5*8):
    plt.subplot(5,8,i+1)
    plt.imshow(images_aug[i], cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])

clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance')
clf.fit(X_train_aug, y_train_aug)
y_pred = clf.predict_proba(X_test)[:,1]
 
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

figsize(6,6)
plot(fpr, tpr)
title('ROC (area under curve=%.3f)' % (metrics.roc_auc_score(y_test, y_pred)))
xlabel('FPR')
ylabel('TPR')
grid(True)

from sklearn import ensemble, linear_model
clf = ensemble.ExtraTreesClassifier(n_estimators=100)  # replace this with any sklearn classifier
clf.fit(X_train_aug, y_train_aug)
y_pred = clf.predict_proba(X_test)[:,1]
 
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

figsize(6,6)
plot(fpr, tpr)
title('ROC (area under curve=%.3f)' % (metrics.roc_auc_score(y_test, y_pred)))
xlabel('FPR')
ylabel('TPR')
grid(True)



