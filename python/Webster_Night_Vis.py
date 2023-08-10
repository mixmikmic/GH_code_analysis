from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import pandas as pd
import random, re

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

labels_path = 'test_files/s_exp_test.csv'
features_path = 'test_files/s_exp_features_test.csv'

s_exp_features = pd.read_csv(features_path, index_col = 0)
s_exp_features.columns = ["Number of Notes", "Location of First Note", "Total Duration of Rests", "Average Max Slope", "Order of Contour", "Consonance"]
s_exp_features.head()

s_exp_features.plot.scatter("Number of Notes", "Location of First Note")

s_exp_features.plot.scatter("Location of First Note", "Total Duration of Rests")

s_exp_features.plot.scatter("Number of Notes", "Average Max Slope")

_input_path = "midi_to_csv/WebsterNight_chords.csv"
_output_path = "generated_solos/12_bar_simple.csv"

a = "C#4"
a[:-1]

_input = pd.read_csv(_input_path).drop("chord",1).drop("velocity",1)
_input["pitch_class"] = [a[:-1] for a in _input["note_name"]]
_input.head()

_output = pd.read_csv(_output_path)
_output["pitch_class"] = [a[:-1] for a in _output["note_name"]]
_output.head()

_input_note_name_prop = _input["note_name"].value_counts()/sum(_input["note_name"].value_counts())
x1 = _input_note_name_prop.plot.bar()
x1.set_xlabel("Pitch")
x1.set_ylabel("Frequency")

_output_note_name_prop = _output["note_name"].value_counts()/sum(_output["note_name"].value_counts())
x2 = _output_note_name_prop.plot.bar()
x2.set_xlabel("Pitch")
x2.set_ylabel("Frequency")

_input_prop_df = pd.DataFrame(_input_note_name_prop).reset_index()
_output_prop_df = pd.DataFrame(_output_note_name_prop).reset_index()
joined_note_name = _input_prop_df.merge(_output_prop_df, on = "index", how = "outer").fillna(0)
joined_note_name.columns = ["note_name", "input","output"]
x1x2 = joined_note_name.plot.bar(x = "note_name")
x1x2.set_xlabel("Note")
x1x2.set_ylabel("Frequency")

_input_pitch_class_prop = _input["pitch_class"].value_counts()/sum(_input["pitch_class"].value_counts())
x1 = _input_pitch_class_prop.plot.bar()
x1.set_xlabel("Pitch Class")
x1.set_ylabel("Frequency")

_output_pitch_class_prop = _output["pitch_class"].value_counts()/sum(_output["pitch_class"].value_counts())
x2 = _output_pitch_class_prop.plot.bar()
x2.set_xlabel("Pitch Class")
x2.set_ylabel("Frequency")

_input_prop_df = pd.DataFrame(_input_pitch_class_prop).reset_index()
_output_prop_df = pd.DataFrame(_output_pitch_class_prop).reset_index()
joined_note_name = _input_prop_df.merge(_output_prop_df, on = "index", how = "outer").fillna(0)
joined_note_name.columns = ["pitch_class", "input","output"]
x1x2 = joined_note_name.plot.bar(x = "pitch_class")
x1x2.set_xlabel("Pitch Class")
x1x2.set_ylabel("Frequency")

x1 = _input["duration"].plot.hist()
x1.set_xlabel("Duration (Fraction of a Measure)")

x2 = _output["duration"].plot.hist()
x2.set_xlabel("Duration (Fraction of a Measure)")

x1 = _input["duration"]
x2 = _output["duration"]

plt.hist(x1, normed = True, alpha = 0.5, label = 'input') #, bins, alpha=0.5, label='x')
plt.hist(x2, normed = True, alpha = 0.5, label = 'output') #, bins, alpha=0.5, label='y')
plt.legend(loc='upper right')
plt.xlabel("Duration (Fraction of a Measure)")
plt.ylabel("Frequency (Normalized)")
plt.show()



