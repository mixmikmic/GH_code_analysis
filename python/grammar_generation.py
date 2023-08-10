import numpy as np
import re

chord_dictionary = {
    "major": {"C": [0, 4, 7], "L": [2, 4, 6, 11]},
    "minor": {"C": [0, 4, 8], "L": [2, 3, 5, 7, 10]},
    "augmented": {"C": [0, 4, 8], "L": []},
    "diminished": {"C": [0, 3, 6] ,  "L": []},
    "half-diminished": {"C": [0, 3, 6, 10], "L": []},
    "dominant-seventh": {"C": [0, 4, 7, 10], "L": [] }
}

notes = ['Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A']
note_to_num = {}
for num, note in enumerate(notes):
    note_to_num[note] = num
num_to_note = dict([[v,k] for k,v in note_to_num.items()])
same_note = {'A#':'Bb', 'C#':'Db', 'D#':'Eb', 'F#': 'Gb', 'G#':'Ab'}

#some basic functionality for working with note strings

#splits note into its components
#also checks if the note is valid
def split_note(note):
    assert re.fullmatch('[A-G](#|b)?[0-7]', note) is not None, 'Note not formatted correctly.'
    return note[:-1], int(note[-1])

def shift_note(note, amount):
    # note taken in as string, amount is any integer
    note, octave = split_note(note)
    if note in same_note:
        note = same_note[note]
    new_num = note_to_num[note] + amount
    if new_num > 11:
        octave += 1
    elif new_num < 0:
        octave -= 1
    return num_to_note[(new_num) % 12] + str(octave)

# Chord isn't chord, it is root note 
def find_note_dist(noteorchord1, note2, chord=False):
    #positive if note2 is above noteorchord1, 0 if same
    tot = 0
    note2, octave2 = split_note(note2)
    if note2 in same_note:
        note2 = same_note[note]
    if not chord:
        noteorchord1, octave1 = split_note(noteorchord1)
        if noteorchord1 in same_note:
            noteorchord1 = same_note[note]
        tot += (octave2 - octave1) * 12
    tot += note_to_num[note2] - note_to_num[noteorchord1]
    return tot
        
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

def get_note_category(note, chord, nextchord):
    root = chord[0]
    chord_type = find_chord_type(chord)
    dist = find_note_dist(root, note, True)
    print(dist)
    for chord_type, interval_list in chord_dictionary[chord_type].items():
        if dist in interval_list:
            return chord_type
    if chord == nextchord:
        return "X"
    else:
        return "A"

def get_notes_from_category(chord, category): 
    possible_notes = []
    root = chord[0] + '4'
    #print(root)
    chord_type = find_chord_type(chord)
    intervals = chord_dictionary[chord_type][category]
    for i in intervals: 
        possible_notes.append(shift_note(root, i)[:-1])
    return possible_notes
     
    

#note_dist("C", "Eb5", True)
#note_category("C7", "Am7b5", "Am7b5")
print(get_note_category("D3", "C", "C"))
get_notes_from_category('Gm7b5', 'C')



