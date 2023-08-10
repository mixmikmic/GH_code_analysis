from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import pandas as pd
import random, re

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

_input_path = "midi_to_csv/convert/KennyGarrett_BrotherHubbard-1_FINAL_chords.csv"
_output_path = "generated_solos/Garrett_Generated.csv"

_input = pd.read_csv(_input_path).drop("chord",1).drop("velocity",1)
_input.head()

_output = pd.read_csv(_output_path)
_output.head()

_input_note_name_prop = _input["note_name"].value_counts()/sum(_input["note_name"].value_counts())
x1 = _input_note_name_prop.plot.bar()
x1.set_xlabel("Note")
x1.set_ylabel("Frequency")

_output_note_name_prop = _output["note_name"].value_counts()/sum(_output["note_name"].value_counts())
x2 = _output_note_name_prop.plot.bar()
x2.set_xlabel("Note")
x2.set_ylabel("Frequency")

_input_prop_df = pd.DataFrame(_input_note_name_prop).reset_index()
_output_prop_df = pd.DataFrame(_output_note_name_prop).reset_index()
joined_note_name = _input_prop_df.merge(_output_prop_df, on = "index", how = "outer").fillna(0)
joined_note_name.columns = ["note_name", "input","output"]
x1x2 = joined_note_name.plot.bar(x = "note_name")
x1x2.set_xlabel("Note")
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



