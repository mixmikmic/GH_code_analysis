from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import pandas as pd
import random, re

labels_path = 'test_files/s_exp.csv'
features_path = 'test_files/s_exp_features.csv'

s_exp_labels = pd.read_csv(labels_path, index_col=0)
s_exp_labels.head()

s_exp_features = pd.read_csv(features_path, index_col = 0)
s_exp_features.head()

notes = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
note_to_num = dict([[n, i] for i, n in enumerate(notes)])
num_to_note = dict([[v, k] for k, v in note_to_num.items()])
same_note = {'A#':'Bb', 'C#':'Db', 'D#':'Eb', 'F#': 'Gb', 'G#':'Ab'}

# checks if a note is formatted correctly and splits it into its component parts
def split_note(note):
    assert re.fullmatch('[A-G][#|b]?[0-7]', note) is not None, 'Note \'%s\' not formatted correctly.'%note
    note, octave = note[:-1], int(note[-1])
    if note in same_note:
        note = same_note[note]
    return note, octave

# shifts the note by amount half-steps (possibly negative)
def shift_note(note, amount):
    note, octave = split_note(note)
    new_num = note_to_num[note] + amount
    if new_num > 11:
        octave += 1
    elif new_num < 0:
        octave -= 1
    return num_to_note[(new_num) % 12] + str(octave)

def get_root(chord):
    r = re.findall('^[A-G][#|b]?', chord) 
    assert r is not None, 'Chord \'%s\'does not contain root note'%chord
    return r[0]

# output is positive if note2 is above noteorchord1, 0 if same
def find_note_dist(note_or_chord1, note2, chord=False):
    note1 = '%s0'%get_root(note_or_chord1) if chord else note_or_chord1
    note1, octave1 = split_note(note1)
    note2, octave2 = split_note(note2)
    dist = (octave2 - octave1) * 12 + note_to_num[note2] - note_to_num[note1]
    return dist % 12 if chord else dist

chord_dictionary = {
    "major": {"C": [0, 4, 7], "L": [2, 5, 9, 11]},
    "minor": {"C": [0, 3, 7], "L": [2, 5, 8, 10]},
#     "augmented": {"C": [0, 4, 8], "L": [2]},
    "diminished": {"C": [0, 3, 6, 9] ,  "L": [2]},
    "half-diminished": {"C": [0, 3, 6, 10], "L": [2]},
    "dominant-seventh": {"C": [0, 4, 7, 10], "L": [2, 5]}
}

def find_chord_type(chord):
    if "m7b5" in chord:
        return "half-diminished"
    elif "j7" in chord:
        return "dominant-seventh"
    elif "o" in chord:
        return "diminished"
    elif "m" in chord: 
        return "minor"
    else:
        return "major"

exp_to_cluster_ratio = 4
k_size = int(len(s_exp_features) / exp_to_cluster_ratio)
k_size

def get_kmeans_clusters(s_exp_features):
    features_scaled = preprocessing.scale(s_exp_features)
    #print(features_scaled.sum(axis=0))
    kmeans = KMeans(n_clusters=k_size).fit(features_scaled)
    return kmeans.labels_

get_kmeans_clusters(s_exp_features)

class Node:
    def __init__(self, node_num):
        self.node_num = node_num
        self.s_exp = [] # list of s-exp-ids
        self.cpt = {} # maps from node_num to conditional probability
    
    def add_exp(self, s):
        self.s_exp.append(s)

def generate_nodes(labels):
    node_objects = [Node(i) for i in range(k_size)] #list of nodes for the Markov chain

    for i, label in enumerate(labels):
        cluster_num = label
        node_objects[label].add_exp(i)
    return node_objects

def generate_cpt(node_objects):
    for outer_node in node_objects:
        outer_node_count = 0
        for inner_node in node_objects:
            outer_node.cpt[inner_node.node_num] = 0.0
            for outer_id in outer_node.s_exp:
                for inner_id in inner_node.s_exp:
                    outer_exp, inner_exp = s_exp_labels.loc[outer_id], s_exp_labels.loc[inner_id]
                    if outer_exp['song_id'] == inner_exp['song_id'] and                        inner_exp['song_index'] - outer_exp['song_index'] == 1:
                            outer_node_count += 1
                            outer_node.cpt[inner_node.node_num] += 1
        if outer_node_count:
            outer_node.cpt = {k: (v / outer_node_count) for k, v in outer_node.cpt.items()}

def select_from_weighted_dct(dct):
    rand = random.random() #random value between 0 and 1
    total = 0
    for k, v in dct.items():
        total += v
        if rand <= total: #if running total exceeds probability, that's what you want
            return k

def sequence_s_expressions(n, node_objects):
    s_exp_ids = []
    next_node = node_objects[random.randint(0, len(node_objects))] # random start node
    for i in range(n):
        next_node_num = select_from_weighted_dct(next_node.cpt)
        next_node = node_objects[next_node_num]
        next_s_exp_id = random.choice(next_node.s_exp)
        s_exp_ids.append(next_s_exp_id) #store id in the list
    return s_exp_ids

def get_notes_from_category(chord, category):
    if category == 'H':
        category = random.choice(['C', 'L'])
    possible_notes = []
    root = get_root(chord) + '4'
    chord_type = find_chord_type(chord)
    if category in ['X', 'A']:
        ex = chord_dictionary[chord_type]['C'] + chord_dictionary[chord_type]['L']
        intervals = [i for i in range(12) if i not in ex]
    else:
        intervals = chord_dictionary[chord_type][category]
    for i in intervals: 
        possible_notes.append(shift_note(root, i)[:-1])
    return possible_notes

def select_note(lst, curr, min_s, max_s, approach_flag): # maybe split into two functions?
    '''
    lst: output of the possible notes (letter only, no octave)
    curr: current note ('C4')
    min_s: minimum slope
    max_s: maximum slope
    
    return: single full note from slope-filtered list; else note from unfiltered list
    '''
    assert lst, "no possible notes!"
    if approach_flag:
        min_s, max_s = max(-1, min_s), min(1, max_s)
    master_lst = []
    octave = split_note(curr)[1]
    all_possible = [note + str(octave + offset) for offset in [-1,0,1] for note in lst]
    for value in all_possible:
        dist = find_note_dist(curr, value)
        if dist >= min_s and dist <= max_s:
            master_lst.append(value)
    if master_lst:
        return random.choice(master_lst)
    return select_note(lst, curr, min_s - 1, max_s + 1, False)

def produce_notes(num_measures, list_of_chords, node_objects): #length of chords list = num of measures
    assert num_measures == len(list_of_chords), "num_measures should be same length as list_of_chords"
    assert num_measures > 0, "num_measures should be positive"
    notes_df = pd.DataFrame(columns=['note_name', 'start_time', 'duration'])
    s_exp_ids = sequence_s_expressions(num_measures, node_objects)
    curr_note = get_root(list_of_chords[0]) + '4'
    approach_flag = False
    for i in range(num_measures): #i refers to measure number
        s_exp = s_exp_labels.loc[s_exp_ids[i], 'exp'] #ith s-exp, a string
        split_list = s_exp.split(' ')
        min_slope, max_slope = int(split_list[0]), int(split_list[1])
        for j, term in enumerate(split_list[2:]):
            elements = term.split("|")
            category, start, duration = elements[0], elements[1], elements[2]
            poss_notes_list = get_notes_from_category(list_of_chords[i], category)
            selected_note = select_note(poss_notes_list, curr_note, min_slope, max_slope, approach_flag)
            new_row = {'note_name': selected_note, 'start_time': float(start) + i, 'duration': duration}
            notes_df = notes_df.append(new_row, ignore_index=True)
            curr_note = selected_note
            approach_flag = category == 'A'
    return notes_df

nodes = generate_nodes(get_kmeans_clusters(s_exp_features))
generate_cpt(nodes)

webster_first8 = ['Em7b5', 'A7', 'Dj7', 'Dj7', 'Em7b5', 'A7', 'Dj7', 'Dj7']
blues_simple12 = ['C7', 'C7', 'C7', 'C7', 'F7', 'F7', 'C7', 'C7', 'G7', 'F7', 'C7', 'G7']
blues_simple12x2 = blues_simple12 + blues_simple12

use_chords = webster_first8
notes_df = produce_notes(len(use_chords), use_chords, nodes)

notes_df['new_duration'] = np.ceil(notes_df['start_time']) - notes_df['start_time']
notes_df['duration'] = notes_df[['duration', 'new_duration']].min(axis=1)
notes_df = notes_df.drop('new_duration', axis=1)

notes_df

# CHANGE FILE NAME BETWEEN GENERATIONS
# notes_df.to_csv('Webster_Generated.csv')

