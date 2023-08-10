from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

mnist_data = input_data.read_data_sets("MNIST_data", one_hot=False)

from sklearn.manifold import t_sne
dim_sne = t_sne.TSNE(n_components=2, n_iter=5000)
lepe_slike = dim_sne.fit_transform(mnist_data.train.images[:10000])

colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'brown', 'silver', 'purple', 'gold']
plt.figure(figsize=(12, 9))
for i in (np.unique(mnist_data.train.labels)):
    for j in range(len(lepe_slike)):
        if mnist_data.train.labels[j] == i:
            plt.scatter(lepe_slike[j, 0], lepe_slike[j, 1], color=colors[i])

#This is faster loop and it is used to create legent for our graph
for i in (np.unique(mnist_data.train.labels)):
       for j in range(len(lepe_slike)):
            if mnist_data.train.labels[j] == i:
                plt.scatter(lepe_slike[j, 0], lepe_slike[j, 1], color=colors[i], label='Image of number {}'.format(i))
                break
plt.legend()
plt.title("t-SNE 2D representation of MNIST dataset")
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

reduced_images = pca.fit_transform(mnist_data.train.images[:10000])

colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'brown', 'silver', 'purple', 'gold']
plt.figure(figsize=(12, 9))
for i in (np.unique(mnist_data.train.labels)):
    for j in range(len(lepe_slike)):
        if mnist_data.train.labels[j] == i:
            plt.scatter(reduced_images[j, 0], reduced_images[j, 1], color=colors[i])

#This is faster loop and it is used to create legent for our graph
for i in (np.unique(mnist_data.train.labels)):
       for j in range(len(lepe_slike)):
            if mnist_data.train.labels[j] == i:
                plt.scatter(reduced_images[j, 0], reduced_images[j, 1], color=colors[i], label='Image of number {}'.format(i))
                break
plt.legend()
plt.title("PCA 2D representation of MNIST dataset")
plt.show()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)

lda_reduction = lda.fit_transform(mnist_data.train.images[:10000], mnist_data.train.labels[:10000])

colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'brown', 'silver', 'purple', 'gold']
plt.figure(figsize=(12, 9))
for i in (np.unique(mnist_data.train.labels)):
    for j in range(len(lepe_slike)):
        if mnist_data.train.labels[j] == i:
            plt.scatter(lda_reduction[j, 0], lda_reduction[j, 1], color=colors[i])

#This is faster loop and it is used to create legent for our graph
for i in (np.unique(mnist_data.train.labels)):
       for j in range(len(lepe_slike)):
            if mnist_data.train.labels[j] == i:
                plt.scatter(lda_reduction[j, 0], lda_reduction[j, 1], color=colors[i], label='Image of number {}'.format(i))
                break
plt.legend()
plt.title("PCA 2D representation of MNIST dataset")
plt.show()

from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf')

reduced_kpca = kpca.fit_transform(mnist_data.train.images[:10000])

colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'brown', 'silver', 'purple', 'gold']
plt.figure(figsize=(12, 9))
for i in (np.unique(mnist_data.train.labels)):
    for j in range(len(lepe_slike)):
        if mnist_data.train.labels[j] == i:
            plt.scatter(reduced_kpca[j, 0], reduced_kpca[j, 1], color=colors[i])

#This is faster loop and it is used to create legent for our graph
for i in (np.unique(mnist_data.train.labels)):
       for j in range(len(lepe_slike)):
            if mnist_data.train.labels[j] == i:
                plt.scatter(reduced_kpca[j, 0], reduced_kpca[j, 1], color=colors[i], label='Image of number {}'.format(i))
                break
plt.legend()
plt.title("PCA 2D representation of MNIST dataset")
plt.show()

