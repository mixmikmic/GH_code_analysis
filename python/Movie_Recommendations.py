get_ipython().magic('matplotlib inline')

import os
import numpy as np

import psycopg2
import psycopg2.extras
from itertools import chain
from collections import Counter, defaultdict
import requests
import imageio

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
from hashlib import md5
from IPython.display import display, HTML

from itertools import chain
import mwparserfromhell

from keras.models import Model
from keras.layers import Embedding, Dense, Input, Lambda, Reshape, merge
import keras.backend as K
from keras.layers.merge import Dot

from sklearn.manifold import TSNE

IMAGE_PATH_EN = 'http://upload.wikimedia.org/wikipedia/en/%s/%s/%s'
IMAGE_PATH_COMMONS = 'http://upload.wikimedia.org/wikipedia/commons/%s/%s/%s'
image_cache = 'movie_images'

def fetch_image(image_name):
    if not image_name or image_name.endswith('.tiff'):
        return None
    image_name = image_name.replace(' ', '_')
    if image_name[0].upper() != image_name[0]:
        image_name = image_name.capitalize()
    file_path = os.path.join(image_cache, image_name)
    if os.path.isfile(file_path):
        return image_name
    else:
        m = md5()
        m.update(image_name.encode('utf-8'))
        c = m.hexdigest()
        path = IMAGE_PATH_EN % (c[0], c[0:2], image_name)
        r = requests.get(path)
        if r.status_code == 404:
            path = IMAGE_PATH_COMMONS % (c[0], c[0:2], image_name)
            r = requests.get(path)
            if r.status_code == 404:
                print image_name
                return None
        try:
            image = Image.open(BytesIO(r.content))
        except IOError:
            return None
        except ValueError:
            return None
        image.save(file(file_path, 'w'))
        image.thumbnail((240, 640), Image.ANTIALIAS)
        res = BytesIO()
        if image.mode == 'P':
            image = image.convert('RGB')
        try:
            image.save(res, 'WEBP', quality=15)
        except IOError as err:
            print image_name, err.message
            return None
        return image_name

fetch_image('Suicide Squad (film) Poster.png')

postgres_conn = psycopg2.connect('dbname=douwe user=notebook')
postgres_cursor = postgres_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

print 'Getting top movies...'
postgres_cursor.execute(
    "SELECT wikipedia.*, wikistats.viewcount FROM wikipedia "
    "JOIN wikistats ON wikipedia.title = wikistats.title WHERE wikipedia.infobox = 'film' "
    "ORDER BY wikistats.viewcount DESC limit 10000"
)
print 'done'

movies = []
for movie in postgres_cursor:
    wikicode = mwparserfromhell.parse(movie['wikitext'])
    image = None
    for template in wikicode.filter_templates():
        if template.name.lower().startswith('infobox '):
            for param in template.params:
                key = param.name.strip()
                if key == 'image':
                    image = param.value.strip()
            break
    if image:
        image_name = fetch_image(image)
    movies.append((movie['title'], image_name, [unicode(x.title) for x in wikicode.filter_wikilinks()], movie['viewcount']))

len(movies)

ref_counts = Counter()
for _, _, refs, _ in movies:
    ref_counts.update(refs)
all_refs = [ref for ref, count in ref_counts.items() if count > 1]
ref_to_id = {ref: idx for idx, ref in enumerate(all_refs)}
movie_to_id = {title: idx for idx, (title, _, _, _) in enumerate(movies)}
len(all_refs), len(ref_counts)

refs_movies = defaultdict(list)
for movie, image, refs, viewcounts in movies:
    movie_id = movie_to_id[movie]
    for ref in refs:
        ref_id = ref_to_id.get(ref)
        if ref_id:
            refs_movies[ref_id].append(movie_id)
refs_movies = list(refs_movies.items())
len(refs_movies)

import random
random.seed(5)

def data_generator(refs_movies, negative_ratio=5, yield_movie_pairs=True):
    random.shuffle(refs_movies)
    for ref, movies in refs_movies:
        if yield_movie_pairs:
            if len(movies) < 2: continue
            a, b = random.sample(movies, 2)
        else:
            a = ref
            b = random.choice(movies)
        yield a, b, 1

        seen = set(movies)
        left = negative_ratio
        while left > 0:
            n = random.randrange(len(movie_to_id))
            if not n in seen:
                left -= 1
                seen.add(n)
                yield a, n, -1


def batchify(gen, batch_size):
    ax, bx, lx = [], [], []
    while True:
        for a, b, label in gen():
            ax.append(a)
            bx.append(b)
            lx.append(label)
            if len(ax) > batch_size:
                yield { 'first': np.asarray(ax), 'second': np.asarray(bx)}, np.asarray(lx)
                del ax[:]
                del bx[:]
                del lx[:]

next(batchify(lambda: data_generator(refs_movies), batch_size=10))

N = 20

def model_simple():
    src = Input(name='first', shape=(1,))
    dst = Input(name='second', shape=(1,))
    src_embedding = Embedding(name='src_embedding', input_dim=len(movie_to_id), output_dim=N)(src)
    dst_embedding = Embedding(name='dst_embedding', input_dim=len(movie_to_id), output_dim=N)(dst)
    dot = merge([src_embedding, dst_embedding], mode='cos')
    dot = Reshape((1,))(dot)
    model = Model(inputs=[src, dst], outputs=[dot])
    model.compile(optimizer='nadam', loss='mse')
    return model

model = model_simple()

model.fit_generator(
    batchify(lambda: data_generator(refs_movies, yield_movie_pairs=True), 2048),
    epochs=25,
    steps_per_epoch=3500,
    verbose=2
)

src = model.get_layer('src_embedding')
src_weights = src.get_weights()[0]
lens = np.linalg.norm(src_weights, axis=1)
normalized = (src_weights.T / lens).T
np.linalg.norm(normalized[0]), normalized.shape

def neighbors(movie):
    dists = np.dot(normalized, normalized[movie_to_id[movie]])
    closest = np.argsort(dists)[-10:]
    for c in closest:
        print(c, movies[c][0], dists[c])

neighbors('Star Wars (film)')

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
xy = model.fit_transform(normalized)
xy

plt.scatter(xy[:,0][:200], xy[:,1][:200])
plt.show()

w = 144
h = 220
res = []
sz = 100
sz_1 = sz + 1
taken = [[False] * sz_1 for _ in range(sz_1)]
x_min = xy.T[0].min()
y_min = xy.T[1].min()
x_max = xy.T[0].max()
y_max = xy.T[1].max()
img = Image.new('RGB', (sz_1 * w, sz_1 * h))
drw = ImageDraw.Draw(img)
c1 = 0
c2 = 0
for movie, coo in zip(movies, xy):
    if not movie[1]:
        continue
    poster = Image.open(image_cache + '/' + movie[1])
    poster.thumbnail((w, h), Image.ANTIALIAS)
    x = int(sz * (coo[0] - x_min) / (x_max - x_min))
    y = int(sz * (coo[1] - y_min) / (y_max - y_min))
    if taken[x][y]:
        c1 += 1
        for dx, dy in (-1, 0), (2, 0), (-1, -1), (0, 2):
            x += dx
            y += dy
            if x >= 0 and y >= 0 and x < sz_1 and y < sz_1 and not taken[x][y]:
                break
        else:
            continue
        c2 += 1
    taken[x][y] = True
    x *= w
    y *= h
    drw.rectangle((x, y, x + w, y + h), (50, 50, 50))
    res.append((x, y, movie[1], poster.size[0], poster.size[1]))
    img.paste(poster, (x + (w - poster.size[0]) / 2, y  + (h - poster.size[1]) / 2))

img.save(open('/home/notebook/notebook/poster.png', 'wb'))
    
x_min, y_min, x_max, y_max, c1, c2

cursor = postgres_conn.cursor()
cursor.execute('DROP TABLE IF EXISTS movie_recommender')
cursor.execute('CREATE TABLE movie_recommender ('
               '    wikipedia_id TEXT PRIMARY KEY,'
               '    viewcount INT,'
               '    image TEXT,'
               '    x FLOAT,'
               '    y FLOAT,'
               '    vec FLOAT[] NOT NULL DEFAULT \'{}\''
               ')')
cursor.execute('CREATE INDEX movie_recommender_vec ON movie_recommender USING gin(vec)')
cursor.execute('CREATE INDEX movie_recommender_name_pattern ON movie_recommender USING btree(lower(wikipedia_id) text_pattern_ops)')
cursor.execute('CREATE INDEX movie_recommender_viewcount ON movie_recommender(viewcount)')

for movie, coo, weights in zip(movies, xy, src_weights):
    x = int(sz * (coo[0] - x_min) / (x_max - x_min)) * w
    y = int(sz * (coo[1] - y_min) / (y_max - y_min)) * h
    v_len = np.linalg.norm(weights)
    cursor.execute('INSERT INTO movie_recommender (wikipedia_id, image, viewcount, x, y, vec) '
                   'VALUES (%s, %s, %s, %s, %s, %s)',
                            (movie[0], movie[1], movie[-1], x, y,
                             [float(weight) / v_len for weight in weights]))
    

postgres_conn.commit()
cursor.close()

neighbors('Star Wars (film)')

coo = xy[639]
x = int(sz * (coo[0] - x_min) / (x_max - x_min)) * w
y = int(sz * (coo[1] - y_min) / (y_max - y_min)) * h
x, y

frames = []
size = 4800
i = 0
x1 = x + 75
y1 = y + 200
while size > 480:
    width2 = int(size / 2)
    height2 = int(size / 3)
    img_crop = img.crop((x1 - width2, y1 - height2, x1 + width2, y1 + height2))
    img_crop = img_crop.resize((600, 400))
    fn = 'movie_images/frame_%d.png' % i
    img_crop.save(fn)
    frames.append(fn)
    size /= 1.2
    i += 1
len(frames)

imageio.mimsave('movie_recommend.gif', [imageio.imread(frame) for frame in frames], 'GIF', duration=0.5)
display(HTML('<img src="movie_recommend.gif">'))

