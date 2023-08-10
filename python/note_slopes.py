import numpy as np
import pandas as pd
import re, os, random

notes = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
note_to_num = {}
for num, note in enumerate(notes):
    note_to_num[note] = num
num_to_note = dict([[v,k] for k,v in note_to_num.items()])
same_note = {'A#':'Bb', 'C#':'Db', 'D#':'Eb', 'F#': 'Gb', 'G#':'Ab'}

def split_note(note):
    assert re.fullmatch('[A-G](#|b)?[0-7]', note) is not None, 'Note not formatted correctly: %s'%note
    return note[:-1], int(note[-1])

def shift_note(note, amount):
    # note taken in as string, amount is any integer
    # probably not needed until actually generating stuff
    note, octave = split_note(note)
    if note in same_note:
        note = same_note[note]
    new_num = note_to_num[note] + amount
    if new_num > 11:
        octave += 1
    elif new_num < 0:
        octave -= 1
    return num_to_note[(new_num) % 12] + str(octave)

def note_dist(note1, note2):
    # positive if note2 is above note1, 0 if same
    note1, octave1 = split_note(note1)
    note2, octave2 = split_note(note2)
    if note1 in same_note:
        note1 = same_note[note1]
    if note2 in same_note:
        note2 = same_note[note2]
    tot = (octave2 - octave1) * 12
    tot += note_to_num[note2] - note_to_num[note1]
    return tot

def find_slope_bounds(lst):
    max_jump, min_jump = 0, 0
    for i in range(len(lst) - 1):
        max_jump = max(max_jump, note_dist(lst[i], lst[i+1]))
        min_jump = min(min_jump, note_dist(lst[i], lst[i+1]))
    return str(min_jump) + ' ' + str(max_jump)

def slope_process(notes):
    slopes = []
    ascending = True
    curr_max = 0
    last_note = notes[0][0]
    direction_changes = 0
    for term in notes[1:]:
        note = term[0]
        dist = note_dist(last_note, note)
        if dist > 0:
            if ascending:
                curr_max = max(curr_max, dist)
            else:
                ascending = True
                slopes.append(curr_max)
                curr_max = dist
                direction_changes += 1
        if dist < 0:
            if not ascending:
                curr_max = max(curr_max, -dist)
            else:
                ascending = False
                slopes.append(curr_max)
                curr_max = -dist
                direction_changes += 1
        last_note = note
    slopes.append(curr_max)
    return np.mean(slopes), direction_changes

avg_max_slope = lambda notes: slope_process(notes)[0]
order_contour = lambda notes: slope_process(notes)[1]

# test = [('D4', 0.5), ('G4', 0.25), ('Bb3', 0.4)]
# avg_max_slope(test)


def select_note(lst, curr, minimum, maximum):
    #1: output of the possible notes
    #2: current note ('C4')
    #3: minimum slope
    #4: maximum slope
    #output single note from filtered list; else random of unfiltered list
    master_lst, lower_octave, equal_octave, higher_octave = [], [], [], []
    for key in lst:
        lower_octave.append(split_note(key)[0] + str((split_note(key)[1]) - 1))
        equal_octave.append(split_note(key)[0] + str(split_note(key)[1]))
        higher_octave.append(split_note(key)[0] + str((split_note(key)[1]) + 1))
    for value in lower_octave:
        if note_dist(value, curr) >= minimum:
            master_lst.append(value)
    for value in equal_octave:
        if note_dist(value, curr) >= minimum:
            master_lst.append(value)
        if note_dist(value, curr) <= maximum:
            master_lst.append(value)
    for value in lower_octave:
        if note_dist(value, curr) <= maximum:
            master_lst.append(value)
    if len(master_lst) == 0:
        return random.choice(lst)
    else:
        return random.choice(master_lst)

#TESTS


#key = 'C4'
#print(split_note(key))
#print(split_note(key)[0])
#print(split_note(key)[1])
#value = (split_note(key)[0] + str((split_note(key)[1]) - 1))
#print(value)

#select_note(['G3', 'F#3', 'A3', 'A#3'], 'C4', 3, 3)

def consonance(s_exp):
    total = 0.0
    measure = s_exp.split(' ')[2:-1]
    
    weights = {'R': 0.1, 'C': 0.8, 'L': 0.4, 'X': 0.1, 'A': 0.6, 'H': 0.6}
    for term in measure:
        note_info = term.split('|')
        note = note_info[0]
        duration = note_info[2]
        if note in weights:
            total += weights[note] * float(duration)
    return total #total consonance value for that measure

# test = '0 4 C|0.155|0.125 H|0.725|0.290'
# print(consonance(test))

loc_first = lambda notes: notes[0][1] % 1

tot_rests = lambda notes: 1 - sum([note[1] for note in notes])

def categorize_note(note, chord, last_chord): #dummy function
    return np.random.choice(['C', 'H', 'R'])

def create_s_exp(notes):
    # notes is list of tupes of (note_string, duration)
    s = ''
    notes_only = []
    for note, start, duration, chord, last_chord in notes:
        s += categorize_note(note, chord, last_chord) + '|%.3f|%.3f '%(duration, start % 1)
        notes_only.append(note)
    return find_slope_bounds(notes_only) + ' ' + s

def featurize(args):
    dummy = lambda x: 0
    feature_funcs = [len, loc_first, tot_rests, avg_max_slope, order_contour, consonance]
    arg_num = [0, 0, 0, 0, 0, 1]
    assert len(feature_funcs) == num_features, "Incorrect number of features"
    features = {}
    for i, func in enumerate(feature_funcs):
        features[str(i)] = func(args[arg_num[i]])
    return features

def process_song(filename, s_exp, features):
    measure = 0
    last_chord = None
    curr_s_exp = []
    song = pd.read_csv(directory + filename)
    for i in range(len(song)):
        curr_note = song.iloc[i]
        if measure != int(curr_note['start_time']):
            s = create_s_exp(curr_s_exp)
            row_s = {'exp': s, 'song_id': song_num, 'song_index': measure}
            s_exp = s_exp.append(row_s, ignore_index=True)
            row_f = featurize([curr_s_exp, s])
            features = features.append(row_f, ignore_index=True)
            curr_s_exp = []
            measure = int(curr_note['start_time'])
        curr_s_exp.append((curr_note['note_name'], curr_note['start_time'], curr_note['duration'], curr_note['chord'], last_chord))
        last_chord = curr_note['chord']
    #tail of loop
    s = create_s_exp(curr_s_exp)
    row_s = {'exp': s, 'song_id': song_num, 'song_index': measure}
    s_exp = s_exp.append(row_s, ignore_index=True)
    row_f = featurize([curr_s_exp, s])
    features = features.append(row_f, ignore_index=True)
    return s_exp, features

num_features = 6

s_exp = pd.DataFrame(columns=['exp', 'song_id', 'song_index'])
features = pd.DataFrame(columns=[str(i) for i in range(num_features)])
directory = 'midi_to_csv/' # 'raw_solos/'
song_num = 0
for filename in os.listdir(directory):
    if filename.endswith('chords.csv'):
        s_exp, features = process_song(filename, s_exp, features)
        song_num += 1
song_num

features.to_csv('test_files/s_exp_')

s_exp.head()

s_exp.to_csv('test_files/s_exp_test.csv')

