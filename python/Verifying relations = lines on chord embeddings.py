import music21
import numpy as np
import scipy as sp
import json

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

def transpose_to_all(c):
    out = []
    old = c
    for i in range(12):
        new = old.transpose(7)
        out.append(new)
        old = new
    return out

def transpose_to_some(c, degree):
    out = []
    out.append(c)
    old = c
    for i in range(degree):
        new = old.transpose(7)
        out.append(new)
        old = new
    return out
        
def c_to_strep(c):
    rep = ""
    for i in range(12):
        if i in c.pitchClasses:
            rep += "1"
        else:
            rep += "0"
    return rep

def strep_to_c(strep):
    pcs = []
    for i, v in enumerate(strep):
        if v == '1':
            pcs.append(i)
    return music21.chord.Chord(pcs)

def strep_to_symbol(strep):
    c = strep_to_c(strep)
    return music21.harmony.chordSymbolFigureFromChord(c)

a = music21.chord.Chord('C E G B-')
b = music21.chord.Chord('F A C')
c = music21.chord.Chord('F A- C')

all_a = transpose_to_all(a)
all_b = transpose_to_all(b)
all_c = transpose_to_all(c)

all_a_str = [c_to_strep(c) for c in all_a]
all_b_str = [c_to_strep(c) for c in all_b]
all_c_str = [c_to_strep(c) for c in all_c]

embeddings = np.load('./embeddings_lite.np.npy')
metadata = json.load(open('./metadata_lite.json'))

labels = [m[0].replace("\"","") for m in metadata[1:]]

pairs = list(zip(all_a_str, all_b_str)) + list(zip(all_a_str, all_c_str))

embeddings.shape, len(labels)

differences = []

for a, b in pairs:
    v_a = embeddings[labels.index(a)]
    v_b = embeddings[labels.index(b)]
    difference = v_a - v_b
    differences.append(difference)

dmat = np.array(differences)

dmat.shape

from scipy.spatial import distance

U, s, V = sp.sparse.linalg.svds(dmat, k=6)
U.shape, s.shape, V.shape

V.shape, dmat.shape

for i in range(24):
    print(distance.cosine(V[3], dmat[i]))





