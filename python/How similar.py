import json 
import numpy as np

def compare(x,y):
    dist = np.sqrt(np.sum(np.square(np.subtract(x, y))))
    return (dist)

face_vectors = json.load(open("all_face_vecs.json"))

import itertools
from IPython.display import Image
from IPython.display import display

eds = ['ed_looking_left.jpg',
 'ed_looking_ahead.jpg',
 'ed_in_sf.jpg',
 'ed_looking_right.jpg',
 'ed_profile.jpg',
 'ed_in_san_diego.jpg',
 'ed_in_connemara.jpg',
 'ed_in_cascades.jpg',
 'ed_in_dc.jpg']

# Here is where we generate a list of all possible pairings of images
pairs_of_eds = list(itertools.combinations(range(len(eds)), 2))
ed_scores = []
for p in pairs_of_eds:
    ed_scores.append(compare(face_vectors[eds[p[0]]], face_vectors[eds[p[1]]]))

ed_vecs = []
for k in eds:
    ed_vecs.append(face_vectors[k])
    
def print_summary_stats(scores):
    print("Average distance: " + str(np.average(scores)))
    print("Minimum distance: " + str(np.min(scores)))
    print("Maximum distance: " + str(np.max(scores)))
    return

def show_image_pair(a, b, height=128):
    a_image = Image(filename = "faces/"+ a, height=height)
    b_image = Image(filename = "faces/"+ b)
    display(a_image,b_image, height=height)    
    return

most_dissimilar = pairs_of_eds[np.argmax(ed_scores)]
most_similar = pairs_of_eds[np.argmin(ed_scores)]

print("Most dissimilar:")
show_image_pair(eds[most_dissimilar[0]], eds[most_dissimilar[1]])

print("Most similar:")
show_image_pair(eds[most_similar[0]], eds[most_similar[1]])

print_summary_stats(ed_scores)

ed_vecs = np.asarray(ed_vecs)
ed_average_vec = np.average(ed_vecs, 0)

ed_scores_vs_average = []
for k in eds:
    ed_scores_vs_average.append(compare(face_vectors[k], ed_average_vec))

print_summary_stats(ed_scores_vs_average)

print ("Most average and most atypical Ed:")
closest_to_canonical = eds[np.argmin(ed_scores_vs_average)]
furthest_from_canonical = eds[np.argmax(ed_scores_vs_average)]

show_image_pair(closest_to_canonical, furthest_from_canonical)

not_ed = ['woman_with_curly_hair.jpg',
 'sunglasses_man.jpg',
 'mustache_man.jpg',
 'little_girl.jpg',
 'beard_in_profile.jpg',
 'lady_in_profile.jpg',
 'man_with_hat.jpg',
 'boy_with_glasses.jpg',
 'lady_with_bangs.jpg']

not_ed_vs_ed = []
for k in not_ed:
    not_ed_vs_ed.append(compare(face_vectors[k], ed_average_vec))
    
print_summary_stats(not_ed_vs_ed)

print ("Most like Ed, most unlike Ed:")
closest_to_ed = not_ed[np.argmin(not_ed_vs_ed)]
furthest_from_ed = not_ed[np.argmax(not_ed_vs_ed)]
show_image_pair(closest_to_ed, furthest_from_ed)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd

def import_more_vecs():
    extra_faces = json.load(open("extra_faces.json", "r"))
    vecs = []
    for k in extra_faces:
        vecs.append(extra_faces[k])
        
    return np.asarray(vecs)
    
    
not_ed_vecs = []
for k in not_ed:
    not_ed_vecs.append(face_vectors[k])
    
X = np.concatenate([ed_vecs, not_ed_vecs], axis=0)
feat_col_names = [ 'x'+str(i) for i in range(X.shape[1]) ]
Y = eds.copy()
Y.extend(not_ed)
is_ed = np.concatenate([np.ones(len(eds)), np.zeros(len(not_ed))])

df = pd.DataFrame(X,columns=feat_col_names)
df['image'] = Y
df['is_ed'] = list(map( lambda x: str(x), is_ed))

from sklearn.manifold import TSNE

# We augmented out small set of face vectors with an external set, to try to improve the stability of the t-SNE
df_lots_of_vecs = pd.DataFrame(data=import_more_vecs(), columns=feat_col_names)
df_lots_of_vecs.append(df[feat_col_names])

tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=500)
tsne_results = tsne.fit_transform(df_lots_of_vecs.values)

df_tsne = df.copy()
df_tsne['x-tsne'] = tsne_results[0:df.shape[0],0]
df_tsne['y-tsne'] = tsne_results[0:df.shape[0],1]
df_tsne['image'] = df['image']
df_tsne['is_ed'] = df['is_ed']

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def scatter_faces(df_faces, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()

    x = df_faces['x-tsne']
    y = df_faces['y-tsne']
    x, y = np.atleast_1d(x, y)
    
    artists = []
    
    for i in range(0, df_tsne.shape[0]):
        try:
            image = plt.imread('faces/' + df_faces.iloc[i]['image'])
        except TypeError:
            # Likely already an array...
            pass
        im = OffsetImage(image, zoom=zoom)        
        ab = AnnotationBbox(im, (x[i], y[i]), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
        
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

fig, ax = plt.subplots(figsize=(12,7))


scatter_faces(df_tsne, ax=ax, zoom=0.25)


x = df_tsne['x-tsne']
y = df_tsne['y-tsne']
x, y = np.atleast_1d(x, y)

    
#ax.plot(x, y)
plt.show()
    



