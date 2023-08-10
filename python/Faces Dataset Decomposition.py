get_ipython().magic('matplotlib inline')
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

IMAGE_SHAPE = (64, 64)
N_ROWS = 5
N_COLS = 7
N_COMPONENTS = N_ROWS * N_COLS
RANDOM_STATE = 42

dataset = fetch_olivetti_faces(shuffle=True, random_state=RANDOM_STATE)
faces = dataset.data

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(faces.shape[0], -1)

def plot_gallery(title, images, n_row=N_ROWS, n_col=N_COLS):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)

    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(IMAGE_SHAPE), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
        
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

plot_gallery("Faces", faces_centered[:N_COMPONENTS])

pca = PCA(n_components=N_COMPONENTS, svd_solver='randomized', whiten=True)
pca.fit(faces_centered)

plot_gallery("Faces", pca.components_)

