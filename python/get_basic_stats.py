get_ipython().magic('matplotlib inline')

from fileoperations.fileoperations import get_filenames_in_dir
import os
import json
import pandas
from collections import Counter
import matplotlib.pyplot as plt

txtfolder = os.path.join('..', '..', 'txt')
symbtrnames = get_filenames_in_dir(txtfolder, keyword='*.txt')[2]
symbtrnames = [os.path.splitext(s)[0] for s in symbtrnames if not s[0] == '.'] 

makamlist = []
formlist = []
usullist = []
composerlist = []
for sn in symbtrnames:
    fields = sn.split('--')
    makamlist.append(fields[0])
    formlist.append(fields[1])
    usullist.append(fields[2])
    composerlist.append(fields[4])

# Makam stats
makam_counts = Counter(makamlist)
print '# of unique makams: ' + str(len(makam_counts))

df_makam = pandas.DataFrame.from_dict(makam_counts, orient='index')
df_makam.columns=['counts']
df_makam = df_makam.sort_values('counts', ascending=False)
df_makam[0:20].plot(kind='bar')
plt.show()

# Form stats
form_counts = Counter(formlist)
print '# of unique forms: ' + str(len(form_counts))

df_form = pandas.DataFrame.from_dict(form_counts, orient='index')
df_form.columns=['counts']
df_form = df_form.sort_values('counts', ascending=False)
df_form[0:20].plot(kind='bar')
plt.show()

# Usul stats
usul_counts = Counter(usullist)
print '# of unique usuls: ' + str(len(usul_counts))

df_usul = pandas.DataFrame.from_dict(usul_counts, orient='index')
df_usul.columns=['counts']
df_usul = df_usul.sort_values('counts', ascending=False)
df_usul[0:20].plot(kind='bar')
plt.show()

# Composer stats
composer_counts = Counter(composerlist)
print '# of unique composers: ' + str(len(composer_counts))

df_composer = pandas.DataFrame.from_dict(composer_counts, orient='index')
df_composer.columns=['counts']
df_composer = df_composer.sort_values('counts', ascending=False)
df_composer[0:20].plot(kind='bar')
plt.show()

