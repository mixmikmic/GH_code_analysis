import numpy as np
import pandas as pd

# import csv of data to pandas dataframe 1980 - 2011.
outcome_vars = pd.read_csv("./outcome_data.csv")
# trim data to relevant years
df_rel_years = outcome_vars[outcome_vars["Year"] >= 1982]
# gets rid of post-2008 years which might complicate data (as there is no exact target)
df_outcome_vars = df_rel_years[df_rel_years["Federal Funds Target Rate"].notnull()]

# add change in Fed funds rate column
changes = np.diff(np.array(df_outcome_vars["Federal Funds Target Rate"]))
df_outcome_vars["Change in Fed Funds Target Rate"] = np.concatenate([[None], changes])

from bs4 import BeautifulSoup
import requests
import re
import urllib.request
import os

# generates a dictionary of appropriate transcript paths
# if you already have the text data, set path_to_local_txt to True. 
link_to_file_on_website = False
path_to_local_pdf = False
path_to_local_txt = True

if link_to_file_on_website:
    base_url = "https://www.federalreserve.gov/monetarypolicy/"
if path_to_local_pdf or path_to_local_txt:
    base_directory = "./feddata/"
    
transcript_links = {}
for year in range(1982, 2009): # from 1982 - 2008
    
    if link_to_file_on_website:
        path = "fomchistorical" + str(year) + ".htm"
        html_doc = requests.get(base_url + path)
        soup = BeautifulSoup(html_doc.content, 'html.parser')
        links = soup.find_all("a", string=re.compile('Transcript .*'))
        link_base_url = "https://www.federalreserve.gov"
        transcript_links[str(year)] = [link_base_url + link["href"] for link in links]
        
    elif path_to_local_pdf or path_to_local_txt:
        files = []
        path_to_folder = base_directory + str(year)
        new_files = os.walk(path_to_folder)
        for file in new_files:
            for f in file[2]:
                if path_to_local_pdf:
                    if f[-3:] == "meeting.pdf":
                        files.append(str(file[0]) + "/" + f)
                elif path_to_local_txt:
                    if f[-11:] == "meeting.txt":
                        files.append(str(file[0]) + "/" + f)
        transcript_links[str(year)] = files
    print("Year Complete: ", year)

# for year in transcript_links.keys():
#     if not os.path.exists("./feddata/" + year):
#         os.makedirs("./feddata/" + year)
#     for link in transcript_links[year]:
#         response = urllib.request.urlopen(str(link))
#         name = re.search("[^/]*$", str(link))
#         print(link)
#         with open("./feddata/" + year + "/" + name.group(), 'wb') as f:
#             f.write(response.read())
#         print("file uploaded")

# create list of all paths and sort in increasing order
sorted_transcripts = []
for linkset in transcript_links.values():
    sorted_transcripts += linkset
sorted_transcripts = sorted(sorted_transcripts)
print("Number of Documents", len(sorted_transcripts))

# from nltk.corpus import stopwords
# i = 0
# for f in sorted_transcripts:
#     infile = open(f, 'r')
#     text = infile.readlines()
#     newfile = open(f[:-4] + 'Stop.txt','w')
#     new_text = []
#     for line in text:
#         mod_line = line[:-1].split(" ")
#         new_line = [word for word in mod_line if word.lower() not in stopwords.words('english')]
#         new_string = ""
#         for word in new_line:
#             new_string += " " + word
#         new_string += "\n"
#         new_text += new_string
#     newfile.writelines(new_text)
#     newfile.close()
#     infile.close()
#     i += 1
#     print("File " + str(i) + " of " + str(len(sorted_transcripts)) + " Completed")

# run all the adjusting of variable sorted_transcripts in order
mod_transcripts = []
for link in sorted_transcripts:
    mod_transcripts.append(str(link)[:-4] + "Stop.txt")
sorted_transcripts = mod_transcripts
print("Number of Documents", len(sorted_transcripts))

# i = 0
# for doc in sorted_transcripts:
#     !python porterstemmer.py {doc}
#     i += 1
#     print("File " + str(i) + " of " + str(len(sorted_transcripts)) + " Completed")
#     print(str(int(i * 100 / len(sorted_transcripts)))  + "%")

# run all the adjusting of variable sorted_transcripts in order
mod_transcripts = []
for link in sorted_transcripts:
    mod_transcripts.append(str(link)[:-4] + "stemmed.txt")
sorted_transcripts = mod_transcripts
print("Number of Documents", len(sorted_transcripts))

from sklearn.feature_extraction.text import TfidfVectorizer

input_vectorizer = TfidfVectorizer(input="filename", stop_words=None)
m = input_vectorizer.fit_transform(sorted_transcripts)

print("Number of Word Stem Vectors:", m.shape[1])
print("Shape of Vector Matrix:", m.shape)

# same thing but with no global weighting
from sklearn.feature_extraction.text import CountVectorizer

input_vectorizer_counts = CountVectorizer(input="filename", stop_words=None)
c = input_vectorizer_counts.fit_transform(sorted_transcripts)

print("Number of Word Stem Vectors:", c.shape[1])
print("Shape of Vector Matrix:", c.shape)
# should be same as above

transcript_path_col = []
weighted_word_count_col = []
word_count_col = []
curr_date_row_counter = 0
subgroup_transcript = []
subgroup_wordcount = []
subgroup_wordcount_c = []
i = 0
while i < len(sorted_transcripts):
    f_month = str(int(df_outcome_vars.iloc[curr_date_row_counter, 1]))
    f_day = str(int(df_outcome_vars.iloc[curr_date_row_counter, 2]))
    f_year = str(int(df_outcome_vars.iloc[curr_date_row_counter, 0]))
    fed_date = pd.to_datetime(f_month + "/" + f_day + "/" + f_year)
    
    link = sorted_transcripts[i]
    date = link.rsplit('/', 1)[-1]
    if date[0] == "F":
        month = date[8:10]
        day = date[10:12]
        year = date[4:8]
    else:
        month = date[4:6]
        day = date[6:8]
        year = date[0:4]
    text_date = pd.to_datetime(month + "/" + day + "/" + year)
    if text_date <= fed_date:
        subgroup_wordcount.append(m[i])
        subgroup_transcript.append(link)
        subgroup_wordcount_c.append(c[i])
        i += 1
    else:
        transcript_path_col.append(subgroup_transcript)
        weighted_word_count_col.append(subgroup_wordcount)
        word_count_col.append(subgroup_wordcount_c)
        subgroup_transcript = []
        subgroup_wordcount = []
        subgroup_wordcount_c = []
        curr_date_row_counter += 1
        
transcript_path_col.append(subgroup_transcript)
weighted_word_count_col.append(subgroup_wordcount)
word_count_col.append(subgroup_wordcount_c)

# append two empty lists representing columns
while len(transcript_path_col) != df_outcome_vars.shape[0] and len(weighted_word_count_col) != df_outcome_vars.shape[0] and len(word_count_col) != df_outcome_vars.shape[0]:
    print("appended column at end")
    transcript_path_col.append([])
    weighted_word_count_col.append([]) 
    word_count_col.append([])
print("Should say 'appended column at end' only twice!")

df_outcome_vars["Transcripts"] = transcript_path_col
df_outcome_vars["WeightedWordCount"] = weighted_word_count_col
df_outcome_vars["WordCount"] = word_count_col

# take a general look at data to decide which words seem to vary the most - would involve some table transformations
relevant_words = ['recoveri', 'save', 'continu', 'expect', 'stock', 'profit', 'gain', 'fund', 'resili', 'household', 'indic', 'bear', 'distribut', 'custom', 'incom', 'bull', 'particip', 'oil', 'suppli', 'employ', 'confid', 'bank', 'forecast', 'price', 'foreign', 'tax', 'stagnat', 'headwind', 'debt', 'wage', 'growth', 'unemploy', 'workforc', 'weak', 'geopolit', 'dramat', 'demand', 'labor', 'consum', 'job', 'produc', 'risk', 'polici', 'strong', 'rate', 'global', 'energi', 'corpor', 'deficit', 'supplier', 'exchang', 'commod', 'wealth', 'inflat', 'condit', 'capit', 'market', 'inflationari', 'economi', 'abroad', 'mortgag', 'percent', 'lack', 'crisi', 'lend']

import functools

# first get the indices of the relevant words in our tokenCount arrays (for use below)
feature_names = input_vectorizer.get_feature_names()
rel_word_indices = [feature_names.index(word) for word in relevant_words]

# create normalization function
def norm(v):
    norm = np.linalg.norm(v)
    return v / norm

# now begin process
rel_word_count = []
for lst in weighted_word_count_col:
    if len(lst) == 0:
        rel_word_count.append([])
    else:
        added_list = functools.reduce(np.add, lst)
        averaged_list = added_list / len(lst)
        rel_word_list = np.array([averaged_list.toarray()[0][i] for i in rel_word_indices])
        normalized_rel_word_list = norm(rel_word_list) * 1000
        rel_word_count.append(normalized_rel_word_list)

df_outcome_vars["RelevantWordVector"] = rel_word_count

# get relevant data (transcripts from year of change and change before) from df for train and test
# fed_rates = np.array(df_outcome_vars["Federal Funds Target Rate"])
change_rates = np.array(df_outcome_vars["Change in Fed Funds Target Rate"])
transcripts = np.array(df_outcome_vars["Transcripts"])
changes = []
i = 0
# while i < len(fed_rates) - 1:
#     if fed_rates[i + 1] != fed_rates[i]:
#         changes.append(True)
#         changes.append(True)
#         i += 2
#     else:
#         changes.append(False)
#         i += 1

# changes.append(False)

changes.append(False)
i += 1
while i < len(change_rates) - 1:
    if change_rates[i] != 0 and len(transcripts[i]) != 0:
        changes.append(True)
    else:
        changes.append(False)
    i += 1

changes.append(False)

relevant_data_df = df_outcome_vars[changes]

# pd.set_option('display.max_rows', len(relevant_data_df))
# relevant_data_df.loc[:, ["Year", "Month", "Day", "Transcripts", "Change in Fed Funds Target Rate", "RelevantWordVector"]]
# pd.reset_option('display.max_rows')

reduction_df = relevant_data_df[relevant_data_df["Change in Fed Funds Target Rate"] < 0]

increase_df = relevant_data_df[relevant_data_df["Change in Fed Funds Target Rate"] > 0]

# split into test/train data.
sample_num1 = reduction_df.shape[0]

def random_bool(shape, p=0.5):
    n = np.prod(shape)
    x = np.fromstring(np.random.bytes(n), np.uint8, n)
    return (x < 255 * p).reshape(shape)

sample_gen1 = random_bool(sample_num1, 0.75)

# see how many samples we're taking / what percentage is train data
c = 0
for b in sample_gen1:
    if b:
        c += 1
        
train_df_reduc = reduction_df[sample_gen1]
test_df_reduc = reduction_df[np.invert(sample_gen1)]

print("REDUCTION")
print("Number of Samples", sample_num1)
print("number of train samples", c)
print("proportion of train samples", c / sample_num1)
print("number of test samples", sample_num1 - c)

sample_num2 = increase_df.shape[0]
sample_gen2 = random_bool(sample_num2, 0.75)
c = 0
for b in sample_gen2:
    if b:
        c += 1

train_df_increas = increase_df[sample_gen2]
test_df_increas = increase_df[np.invert(sample_gen2)]

print("INCREASE")
print("Number of Samples", sample_num2)
print("number of train samples", c)
print("proportion of train samples", c / sample_num2)
print("number of test samples", sample_num2 - c)

Y_train_reduc = np.array(train_df_reduc["Change in Fed Funds Target Rate"])
X_train_reduc = [list(x) for x in np.array(train_df_reduc["RelevantWordVector"])]

Y_test_reduc = np.array(test_df_reduc["Change in Fed Funds Target Rate"])
X_test_reduc = [list(x) for x in np.array(test_df_reduc["RelevantWordVector"])]

Y_train_increas = np.array(train_df_increas["Change in Fed Funds Target Rate"])
X_train_increas = [list(x) for x in np.array(train_df_increas["RelevantWordVector"])]

Y_test_increas = np.array(test_df_increas["Change in Fed Funds Target Rate"])
X_test_increas = [list(x) for x in np.array(test_df_increas["RelevantWordVector"])]

import sklearn.linear_model as lm

linear_clf = lm.LassoCV(tol=.001, cv=3)

# Fit your classifier
linear_clf.fit(X_train_reduc, Y_train_reduc)

# Output Coefficients
print(linear_clf.coef_)

linear_clf.coef_

i = 0
while i < len(linear_clf.coef_):
    print((linear_clf.coef_[i], relevant_words[i]))
    i += 1

# Output RMSE on test set
def rmse(predicted_y, actual_y):
    return np.sqrt(np.mean((predicted_y - actual_y) ** 2))

print("RMSE", rmse(linear_clf.predict(X_test_reduc), Y_test_reduc))

linear_clf.predict(X_test_reduc)

Y_test_reduc

t_stats = []
std = []
coefficient_matrix = linear_clf.coef_
i = 0
while i < len(coefficient_matrix):
    vals = []
    for x_sample in X_train_reduc:
        vals.append(x_sample[i])
    standard_deviation = np.std(vals)
    std.append(standard_deviation)
    n = len(X_train_reduc)
    standard_error = standard_deviation / np.sqrt(n)
    t_stats.append((coefficient_matrix[i] - 0) / standard_error)
    i += 1
std = np.array(std)
std= np.round(std, decimals=2)

print(t_stats)
print("\n")
print(std)
print(np.mean(std))

import plotly 
plotly.tools.set_credentials_file(username='ali-wetrill', api_key='x8ZULc5B5lFJLxnFCFD8')

basis_changes = sorted(list(set(list(reduction_df["Change in Fed Funds Target Rate"]))))
basis_changes.reverse()

vals = [[np.sum(x[0]) for x in list(reduction_df[reduction_df["Change in Fed Funds Target Rate"] == j]["WordCount"])] for j in basis_changes]

len(basis_changes)

import plotly.plotly as py
import plotly.graph_objs as go

colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']

trace0 = go.Box(
    y = vals[0],
    name = str(basis_changes[0]),
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'all',
    marker = dict(
        color = colors[0]),
    line = dict(
        color = colors[0])
)

trace1 = go.Box(
    y = vals[1],
    name = str(basis_changes[1]),
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'all',
    marker = dict(
        color = colors[1]),
    line = dict(
        color = colors[1])
)

trace2 = go.Box(
    y = vals[2],
    name = str(basis_changes[2]),
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'all',
    marker = dict(
        color = colors[2]),
    line = dict(
        color = colors[2])
)

trace3 = go.Box(
    y = vals[3],
    name = str(basis_changes[3]),
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'all',
    marker = dict(
        color = colors[3]),
    line = dict(
        color = colors[3])
)

trace4 = go.Box(
    y = vals[4],
    name = str(basis_changes[4]),
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'all',
    marker = dict(
        color = colors[4]),
    line = dict(
        color = colors[4])
)

trace5 = go.Box(
    y = vals[5],
    name = str(basis_changes[5]),
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'all',
    marker = dict(
        color = colors[5]),
    line = dict(
        color = colors[5])
)

data = [trace0,trace1,trace2,trace3,trace4,trace5]

layout = go.Layout(
    title = "Variation in Words within each Transcript",
    xaxis=dict(
        type="category"
    )
)

fig = go.Figure(data=data,layout=layout)
py.iplot(fig, filename = "Box Plot Styling Outliers")

word_count_per_basis_change = []
for rate in basis_changes:
    z = list(reduction_df[reduction_df["Change in Fed Funds Target Rate"] == rate]["WordCount"])
    word_count_per_basis_change.append([sum([x[0].toarray()[0][i] for x in z]) for i in range(0, len(z[0][0].toarray()[0]))])

total_word_count = [sum(word_count_per_basis_change[i]) for i in range(0, len(word_count_per_basis_change))]

percent_word_counts = [np.array(word_count_per_basis_change[i]) / total_word_count[i] for i in range(0, len(total_word_count))]

percent_word_counts_with_words = []
for arr in percent_word_counts:
    i = 0
    new_arr = []
    names = input_vectorizer_counts.get_feature_names()
    while i < len(arr):
        new_arr.append((arr[i], names[i]))
        i += 1
    percent_word_counts_with_words.append(new_arr)

percent_word_counts_with_words = [sorted(list(x)) for x in percent_word_counts_with_words]
dummy = [x.reverse() for x in percent_word_counts_with_words]

# manual inspetion. adjust i for starting basis_changes. j for within a basis_change.
# i = 2
# while i < len(percent_word_counts_with_words):
#     print(basis_changes[i])
#     j = 43
#     while j < 50:
#         print(percent_word_counts_with_words[i][j])
#         j += 1
#     print("\n")
#     i += 1

#manual inspection on which to plot
ind_2_25 = [5, 12, 13, 14, 15, 16, 26, 34, 43, 44]
ind_4_05 = [3, 5, 13, 14, 15, 18, 22, 25, 26, 28]

i = 2
print(basis_changes[i])
for j in ind_2_25:
    val = percent_word_counts_with_words[i][j]
    print(str(np.around(val[0] * 100, decimals=3)) + "%" + " " + val[1])
i = 4
print("\n")
print(basis_changes[i])
for j in ind_4_05:
    val = percent_word_counts_with_words[i][j]
    print(str(np.around(val[0] * 100, decimals=3)) + "%" + " " + val[1])

i = 2
j = 0
ind_relwords_25 = []
while j < len(percent_word_counts_with_words[i]):
    if percent_word_counts_with_words[i][j][1] in relevant_words:
        ind_relwords_25.append(j)
    j += 1

i = 4
j = 0
ind_relwords_05 = []
while j < len(percent_word_counts_with_words[i]):
    if percent_word_counts_with_words[i][j][1] in relevant_words:
        ind_relwords_05.append(j)
    j += 1

i = 2
print(basis_changes[i])
for j in ind_relwords_25:
    val = percent_word_counts_with_words[i][j]
    print(str(np.around(val[0] * 100, decimals=3)) + "%" + " " + val[1])
i = 4
print("\n")
print(basis_changes[i])
for j in ind_relwords_05:
    val = percent_word_counts_with_words[i][j]
    print(str(np.around(val[0] * 100, decimals=3)) + "%" + " " + val[1])

# plot how they change over time

# top word counts in every date
# sum of all words in every date for neg changesn
# list of lists, with each inner list holding a particular word frequency change over time

word_counts_with_words = []
z = 1
for arr in list(reduction_df["WordCount"]):
    lst = arr[0].toarray()[0]
    s = sum(lst)
    i = 0
    new_arr = []
    names = input_vectorizer_counts.get_feature_names()
    while i < len(lst):
        new_arr.append((lst[i] * 100 / s, names[i]))
        i += 1
    word_counts_with_words.append(new_arr)

word_counts_with_words = [sorted(list(x)) for x in word_counts_with_words]
dummy = [x.reverse() for x in word_counts_with_words]

# lst_ind_relwords = []
# i = 0
# while i < len(word_counts_with_words):
#     j = 0
#     ind_relwords = []
#     while j < len(word_counts_with_words[i]):
#         if word_counts_with_words[i][j][1] in relevant_words:
#             ind_relwords.append(j)
#         j += 1
#     lst_ind_relwords.append(ind_relwords)
#     i += 1

rel_word_counts_with_words = []
i = 0
while i < len(word_counts_with_words):
    lst = word_counts_with_words[i]
    new_lst = []
    j = 0
    while j < len(lst):
        if lst[j][1] in relevant_words:
            new_lst.append(lst[j])
        j += 1
    rel_word_counts_with_words.append(new_lst)
    i += 1

def findItem(theList, word):
    return [theList.index(i) for i in theList if i[1] == word][0]

rel_word_g2 = [[findItem(x, word) for x in rel_word_counts_with_words] for word in relevant_words]

# Relevant Lists
# rel_word_counts_with_words - word counts for each date. In each list use rel_word_g2 to get word count.
# rel_word_g2 - for each index, lst is where that word is in the above list.
# relevant_words - word.

rel_word_counts_with_words[0][16]

y_vals = []
for word_indices in rel_word_g2:
    i = 0
    sub_list = []
    while i < len(rel_word_counts_with_words):
        index = word_indices[i]
        date_lst = rel_word_counts_with_words[i]
        sub_list.append(date_lst[index])
        i += 1
    y_vals.append(sub_list)

labels = relevant_words

yr = list(reduction_df["Year"])
mnth = list(reduction_df["Month"])
day = list(reduction_df["Day"])
x_vals = []
for i in zip(mnth, day, yr):
    date = pd.to_datetime(str(i[0]) + "/" + str(i[1]) + "/" + str(i[2]))
    x_vals.append(date)

import plotly.plotly as py
import plotly.graph_objs as go

# Create random data with numpy
import numpy as np

# Create traces
trace0 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[0]],
    mode = 'lines',
    name = relevant_words[0]
)

trace1 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[1]],
    mode = 'lines',
    name = relevant_words[1]
)

trace2 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[2]],
    mode = 'lines',
    name = relevant_words[2]
)

trace3 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[3]],
    mode = 'lines',
    name = relevant_words[3]
)

trace4 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[4]],
    mode = 'lines',
    name = relevant_words[4]
)

trace5 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[5]],
    mode = 'lines',
    name = relevant_words[5]
)

trace6 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[6]],
    mode = 'lines',
    name = relevant_words[6]
)

trace7 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[7]],
    mode = 'lines',
    name = relevant_words[7]
)

trace8 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[8]],
    mode = 'lines',
    name = relevant_words[8]
)

trace9 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[9]],
    mode = 'lines',
    name = relevant_words[9]
)

trace10 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[10]],
    mode = 'lines',
    name = relevant_words[10]
)

trace11 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[11]],
    mode = 'lines',
    name = relevant_words[11]
)

trace12 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[12]],
    mode = 'lines',
    name = relevant_words[12]
)

trace13 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[13]],
    mode = 'lines',
    name = relevant_words[13]
)

trace14 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[14]],
    mode = 'lines',
    name = relevant_words[14]
)

trace15 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[15]],
    mode = 'lines',
    name = relevant_words[15]
)

trace16 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[16]],
    mode = 'lines',
    name = relevant_words[16]
)

trace17 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[17]],
    mode = 'lines',
    name = relevant_words[17]
)

trace18 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[18]],
    mode = 'lines',
    name = relevant_words[18]
)

trace19 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[19]],
    mode = 'lines',
    name = relevant_words[19]
)

trace20 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[20]],
    mode = 'lines',
    name = relevant_words[20]
)

trace21 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[21]],
    mode = 'lines',
    name = relevant_words[21]
)

trace22 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[22]],
    mode = 'lines',
    name = relevant_words[22]
)

trace23 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[23]],
    mode = 'lines',
    name = relevant_words[23]
)

trace24 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[24]],
    mode = 'lines',
    name = relevant_words[24]
)

trace25 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[25]],
    mode = 'lines',
    name = relevant_words[25]
)

trace26 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[26]],
    mode = 'lines',
    name = relevant_words[26]
)

trace27 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[27]],
    mode = 'lines',
    name = relevant_words[27]
)

trace28 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[28]],
    mode = 'lines',
    name = relevant_words[28]
)

trace29 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[29]],
    mode = 'lines',
    name = relevant_words[29]
)

trace30 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[30]],
    mode = 'lines',
    name = relevant_words[30]
)

trace31 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[31]],
    mode = 'lines',
    name = relevant_words[31]
)

trace32 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[32]],
    mode = 'lines',
    name = relevant_words[32]
)

trace33 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[33]],
    mode = 'lines',
    name = relevant_words[33]
)

trace34 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[34]],
    mode = 'lines',
    name = relevant_words[34]
)

trace35 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[35]],
    mode = 'lines',
    name = relevant_words[35]
)

trace36 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[36]],
    mode = 'lines',
    name = relevant_words[36]
)

trace37 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[37]],
    mode = 'lines',
    name = relevant_words[37]
)

trace38 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[38]],
    mode = 'lines',
    name = relevant_words[38]
)

trace39 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[39]],
    mode = 'lines',
    name = relevant_words[39]
)

trace40 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[40]],
    mode = 'lines',
    name = relevant_words[40]
)

trace41 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[41]],
    mode = 'lines',
    name = relevant_words[41]
)

trace42 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[42]],
    mode = 'lines',
    name = relevant_words[42]
)

trace43 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[43]],
    mode = 'lines',
    name = relevant_words[43]
)

trace44 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[44]],
    mode = 'lines',
    name = relevant_words[44]
)

trace45 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[45]],
    mode = 'lines',
    name = relevant_words[45]
)

trace46 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[46]],
    mode = 'lines',
    name = relevant_words[46]
)

trace47 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[47]],
    mode = 'lines',
    name = relevant_words[47]
)

trace48 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[48]],
    mode = 'lines',
    name = relevant_words[48]
)

trace49 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[49]],
    mode = 'lines',
    name = relevant_words[49]
)

trace50 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[50]],
    mode = 'lines',
    name = relevant_words[50]
)

trace51 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[51]],
    mode = 'lines',
    name = relevant_words[51]
)

trace52 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[52]],
    mode = 'lines',
    name = relevant_words[52]
)

trace53 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[53]],
    mode = 'lines',
    name = relevant_words[53]
)

trace54 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[54]],
    mode = 'lines',
    name = relevant_words[54]
)

trace55 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[55]],
    mode = 'lines',
    name = relevant_words[55]
)

trace56 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[56]],
    mode = 'lines',
    name = relevant_words[56]
)

trace57 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[57]],
    mode = 'lines',
    name = relevant_words[57]
)

trace58 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[58]],
    mode = 'lines',
    name = relevant_words[58]
)

trace59 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[59]],
    mode = 'lines',
    name = relevant_words[59]
)

trace60 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[60]],
    mode = 'lines',
    name = relevant_words[60]
)

trace61 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[61]],
    mode = 'lines',
    name = relevant_words[61]
)

trace62 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[62]],
    mode = 'lines',
    name = relevant_words[62]
)

trace63 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[63]],
    mode = 'lines',
    name = relevant_words[63]
)

trace64 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[64]],
    mode = 'lines',
    name = relevant_words[64]
)

data = [
       trace2, trace3, trace7, 
       trace21, trace22, trace23, 
       trace30,
       trace41, trace42,
       trace53
]

# data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, 
#         trace10, trace11, trace12, trace13, trace14, trace15, trace16, trace17, trace18, trace19,
#        trace20, trace21, trace22, trace23, trace24, trace25, trace26, trace27, trace28, trace29, 
#        trace30, trace31, trace32, trace33, trace34, trace35, trace36, trace37, trace38, trace39,
#        trace40, trace41, trace42, trace43, trace44, trace45, trace46, trace47, trace48, trace49,
#        trace50, trace51, trace52, trace53, trace54, trace55, trace56, trace57, trace58, trace59,
#        trace60, trace61, trace62, trace63, trace64]

py.iplot(data, filename='line-mode')

# relevant_words.index("economi")

# ind = [2, 3, 7, 21, 22, 23, 30, 41, 42, 44, 53, 56, 58, 61]
# avg = []
# for i in ind:
#     avg.append(np.mean([y[0] for y in y_vals[i]]))

import sklearn.linear_model as lm

linear_clf2 = lm.LassoCV(tol=.001, cv=3)

# Fit your classifier
linear_clf2.fit(X_train_increas, Y_train_increas)

# Output Coefficients
print(linear_clf2.coef_)

i = 0
while i < len(linear_clf2.coef_):
    print((linear_clf2.coef_[i], relevant_words[i]))
    i += 1

# Output RMSE on test set
def rmse(predicted_y, actual_y):
    return np.sqrt(np.mean((predicted_y - actual_y) ** 2))

print("RMSE", rmse(linear_clf2.predict(X_test_increas), Y_test_increas))

linear_clf2.predict(X_test_increas)

Y_test_increas

t_stats = []
std = []
coefficient_matrix = linear_clf2.coef_
i = 0
while i < len(coefficient_matrix):
    vals = []
    for x_sample in X_train_increas:
        vals.append(x_sample[i])
    standard_deviation = np.std(vals)
    std.append(standard_deviation)
    n = len(X_train_increas)
    standard_error = standard_deviation / np.sqrt(n)
    t_stats.append((coefficient_matrix[i] - 0) / standard_error)
    i += 1
std = np.array(std)
std= np.round(std, decimals=2)

print(t_stats)
print("\n")
print(std)
print(np.mean(std))

basis_changes = sorted(list(set(list(increase_df["Change in Fed Funds Target Rate"]))))
basis_changes.reverse()

basis_changes

vals = [[np.sum(x[0]) for x in list(increase_df[increase_df["Change in Fed Funds Target Rate"] == j]["WordCount"])] for j in basis_changes]

import plotly.plotly as py
import plotly.graph_objs as go

colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']

trace0 = go.Box(
    y = vals[0],
    name = str(basis_changes[0]),
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'all',
    marker = dict(
        color = colors[0]),
    line = dict(
        color = colors[0])
)

trace1 = go.Box(
    y = vals[1],
    name = str(basis_changes[1]),
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'all',
    marker = dict(
        color = colors[1]),
    line = dict(
        color = colors[1])
)

trace2 = go.Box(
    y = vals[2],
    name = str(basis_changes[2]),
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'all',
    marker = dict(
        color = colors[2]),
    line = dict(
        color = colors[2])
)

trace3 = go.Box(
    y = vals[3],
    name = str(basis_changes[3]),
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'all',
    marker = dict(
        color = colors[3]),
    line = dict(
        color = colors[3])
)

trace4 = go.Box(
    y = vals[4],
    name = str(basis_changes[4]),
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'all',
    marker = dict(
        color = colors[4]),
    line = dict(
        color = colors[4])
)

trace5 = go.Box(
    y = vals[5],
    name = str(basis_changes[5]),
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'all',
    marker = dict(
        color = colors[5]),
    line = dict(
        color = colors[5])
)

data = [trace0,trace1,trace2,trace3,trace4,trace5]

layout = go.Layout(
    title = "Variation in Words within each Transcript",
    xaxis=dict(
        type="category"
    )
)

fig = go.Figure(data=data,layout=layout)
py.iplot(fig, filename = "Box Plot Styling Outliers")

word_count_per_basis_change = []
for rate in basis_changes:
    z = list(increase_df[increase_df["Change in Fed Funds Target Rate"] == rate]["WordCount"])
    word_count_per_basis_change.append([sum([x[0].toarray()[0][i] for x in z]) for i in range(0, len(z[0][0].toarray()[0]))])

total_word_count = [sum(word_count_per_basis_change[i]) for i in range(0, len(word_count_per_basis_change))]

percent_word_counts = [np.array(word_count_per_basis_change[i]) / total_word_count[i] for i in range(0, len(total_word_count))]

percent_word_counts_with_words = []
for arr in percent_word_counts:
    i = 0
    new_arr = []
    names = input_vectorizer_counts.get_feature_names()
    while i < len(arr):
        new_arr.append((arr[i], names[i]))
        i += 1
    percent_word_counts_with_words.append(new_arr)

percent_word_counts_with_words = [sorted(list(x)) for x in percent_word_counts_with_words]
dummy = [x.reverse() for x in percent_word_counts_with_words]

# manual inspection. adjust i for starting basis_changes. j for within a basis_change.
# i = 5
# while i < len(percent_word_counts_with_words):
#     print(basis_changes[i])
#     j = 33
#     while j < 50:
#         print(percent_word_counts_with_words[i][j])
#         j += 1
#     print("\n")
#     break
#     i += 1

#manual inspection on which to plot
ind_2_05 = [4, 6, 10, 14, 17, 18, 19, 21, 27, 29]
ind_4_25 = [2, 5, 6, 7, 8, 9, 11, 16, 17, 18]
ind_5_125 = [6, 17, 21, 24, 27, 28, 30, 32, 33]

i = 2
print(basis_changes[i])
for j in ind_2_05:
    val = percent_word_counts_with_words[i][j]
    print(str(np.around(val[0] * 100, decimals=3)) + "%" + " " + val[1])
i = 4
print("\n")
print(basis_changes[i])
for j in ind_4_25:
    val = percent_word_counts_with_words[i][j]
    print(str(np.around(val[0] * 100, decimals=3)) + "%" + " " + val[1])
i = 5
print("\n")
print(basis_changes[i])
for j in ind_5_125:
    val = percent_word_counts_with_words[i][j]
    print(str(np.around(val[0] * 100, decimals=3)) + "%" + " " + val[1])

# a closer look at relevant words

i = 2
j = 0
ind_relwords_05 = []
while j < len(percent_word_counts_with_words[i]):
    if percent_word_counts_with_words[i][j][1] in relevant_words:
        ind_relwords_05.append(j)
    j += 1

i = 4
j = 0
ind_relwords_25 = []
while j < len(percent_word_counts_with_words[i]):
    if percent_word_counts_with_words[i][j][1] in relevant_words:
        ind_relwords_25.append(j)
    j += 1

i = 5
j = 0
ind_relwords_125 = []
while j < len(percent_word_counts_with_words[i]):
    if percent_word_counts_with_words[i][j][1] in relevant_words:
        ind_relwords_125.append(j)
    j += 1

i = 2
print(basis_changes[i])
for j in ind_relwords_05:
    val = percent_word_counts_with_words[i][j]
    print(str(np.around(val[0] * 100, decimals=3)) + "%" + " " + val[1])
i = 4
print("\n")
print(basis_changes[i])
for j in ind_relwords_25:
    val = percent_word_counts_with_words[i][j]
    print(str(np.around(val[0] * 100, decimals=3)) + "%" + " " + val[1])
i = 5
print("\n")
print(basis_changes[i])
for j in ind_relwords_125:
    val = percent_word_counts_with_words[i][j]
    print(str(np.around(val[0] * 100, decimals=3)) + "%" + " " + val[1])

# plot how they change over time

word_counts_with_words = []
z = 1
for arr in list(increase_df["WordCount"]):
    lst = arr[0].toarray()[0]
    s = sum(lst)
    i = 0
    new_arr = []
    names = input_vectorizer_counts.get_feature_names()
    while i < len(lst):
        new_arr.append((lst[i] * 100 / s, names[i]))
        i += 1
    word_counts_with_words.append(new_arr)

word_counts_with_words = [sorted(list(x)) for x in word_counts_with_words]
dummy = [x.reverse() for x in word_counts_with_words]

rel_word_counts_with_words = []
i = 0
while i < len(word_counts_with_words):
    lst = word_counts_with_words[i]
    new_lst = []
    j = 0
    while j < len(lst):
        if lst[j][1] in relevant_words:
            new_lst.append(lst[j])
        j += 1
    rel_word_counts_with_words.append(new_lst)
    i += 1

def findItem(theList, word):
    return [theList.index(i) for i in theList if i[1] == word][0]

rel_word_g2 = [[findItem(x, word) for x in rel_word_counts_with_words] for word in relevant_words]

y_vals = []
for word_indices in rel_word_g2:
    i = 0
    sub_list = []
    while i < len(rel_word_counts_with_words):
        index = word_indices[i]
        date_lst = rel_word_counts_with_words[i]
        sub_list.append(date_lst[index])
        i += 1
    y_vals.append(sub_list)

labels = relevant_words

yr = list(increase_df["Year"])
mnth = list(increase_df["Month"])
day = list(increase_df["Day"])
x_vals = []
for i in zip(mnth, day, yr):
    date = pd.to_datetime(str(i[0]) + "/" + str(i[1]) + "/" + str(i[2]))
    x_vals.append(date)

import plotly.plotly as py
import plotly.graph_objs as go

# Create random data with numpy
import numpy as np

# Create traces
trace0 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[0]],
    mode = 'lines',
    name = relevant_words[0]
)

trace1 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[1]],
    mode = 'lines',
    name = relevant_words[1]
)

trace2 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[2]],
    mode = 'lines',
    name = relevant_words[2]
)

trace3 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[3]],
    mode = 'lines',
    name = relevant_words[3]
)

trace4 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[4]],
    mode = 'lines',
    name = relevant_words[4]
)

trace5 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[5]],
    mode = 'lines',
    name = relevant_words[5]
)

trace6 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[6]],
    mode = 'lines',
    name = relevant_words[6]
)

trace7 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[7]],
    mode = 'lines',
    name = relevant_words[7]
)

trace8 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[8]],
    mode = 'lines',
    name = relevant_words[8]
)

trace9 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[9]],
    mode = 'lines',
    name = relevant_words[9]
)

trace10 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[10]],
    mode = 'lines',
    name = relevant_words[10]
)

trace11 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[11]],
    mode = 'lines',
    name = relevant_words[11]
)

trace12 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[12]],
    mode = 'lines',
    name = relevant_words[12]
)

trace13 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[13]],
    mode = 'lines',
    name = relevant_words[13]
)

trace14 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[14]],
    mode = 'lines',
    name = relevant_words[14]
)

trace15 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[15]],
    mode = 'lines',
    name = relevant_words[15]
)

trace16 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[16]],
    mode = 'lines',
    name = relevant_words[16]
)

trace17 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[17]],
    mode = 'lines',
    name = relevant_words[17]
)

trace18 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[18]],
    mode = 'lines',
    name = relevant_words[18]
)

trace19 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[19]],
    mode = 'lines',
    name = relevant_words[19]
)

trace20 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[20]],
    mode = 'lines',
    name = relevant_words[20]
)

trace21 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[21]],
    mode = 'lines',
    name = relevant_words[21]
)

trace22 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[22]],
    mode = 'lines',
    name = relevant_words[22]
)

trace23 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[23]],
    mode = 'lines',
    name = relevant_words[23]
)

trace24 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[24]],
    mode = 'lines',
    name = relevant_words[24]
)

trace25 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[25]],
    mode = 'lines',
    name = relevant_words[25]
)

trace26 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[26]],
    mode = 'lines',
    name = relevant_words[26]
)

trace27 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[27]],
    mode = 'lines',
    name = relevant_words[27]
)

trace28 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[28]],
    mode = 'lines',
    name = relevant_words[28]
)

trace29 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[29]],
    mode = 'lines',
    name = relevant_words[29]
)

trace30 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[30]],
    mode = 'lines',
    name = relevant_words[30]
)

trace31 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[31]],
    mode = 'lines',
    name = relevant_words[31]
)

trace32 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[32]],
    mode = 'lines',
    name = relevant_words[32]
)

trace33 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[33]],
    mode = 'lines',
    name = relevant_words[33]
)

trace34 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[34]],
    mode = 'lines',
    name = relevant_words[34]
)

trace35 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[35]],
    mode = 'lines',
    name = relevant_words[35]
)

trace36 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[36]],
    mode = 'lines',
    name = relevant_words[36]
)

trace37 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[37]],
    mode = 'lines',
    name = relevant_words[37]
)

trace38 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[38]],
    mode = 'lines',
    name = relevant_words[38]
)

trace39 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[39]],
    mode = 'lines',
    name = relevant_words[39]
)

trace40 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[40]],
    mode = 'lines',
    name = relevant_words[40]
)

trace41 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[41]],
    mode = 'lines',
    name = relevant_words[41]
)

trace42 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[42]],
    mode = 'lines',
    name = relevant_words[42]
)

trace43 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[43]],
    mode = 'lines',
    name = relevant_words[43]
)

trace44 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[44]],
    mode = 'lines',
    name = relevant_words[44]
)

trace45 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[45]],
    mode = 'lines',
    name = relevant_words[45]
)

trace46 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[46]],
    mode = 'lines',
    name = relevant_words[46]
)

trace47 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[47]],
    mode = 'lines',
    name = relevant_words[47]
)

trace48 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[48]],
    mode = 'lines',
    name = relevant_words[48]
)

trace49 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[49]],
    mode = 'lines',
    name = relevant_words[49]
)

trace50 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[50]],
    mode = 'lines',
    name = relevant_words[50]
)

trace51 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[51]],
    mode = 'lines',
    name = relevant_words[51]
)

trace52 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[52]],
    mode = 'lines',
    name = relevant_words[52]
)

trace53 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[53]],
    mode = 'lines',
    name = relevant_words[53]
)

trace54 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[54]],
    mode = 'lines',
    name = relevant_words[54]
)

trace55 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[55]],
    mode = 'lines',
    name = relevant_words[55]
)

trace56 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[56]],
    mode = 'lines',
    name = relevant_words[56]
)

trace57 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[57]],
    mode = 'lines',
    name = relevant_words[57]
)

trace58 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[58]],
    mode = 'lines',
    name = relevant_words[58]
)

trace59 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[59]],
    mode = 'lines',
    name = relevant_words[59]
)

trace60 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[60]],
    mode = 'lines',
    name = relevant_words[60]
)

trace61 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[61]],
    mode = 'lines',
    name = relevant_words[61]
)

trace62 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[62]],
    mode = 'lines',
    name = relevant_words[62]
)

trace63 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[63]],
    mode = 'lines',
    name = relevant_words[63]
)

trace64 = go.Scatter(
    x = x_vals,
    y = [y[0] for y in y_vals[64]],
    mode = 'lines',
    name = relevant_words[64]
)

data = [trace2, trace3, trace7, 
        trace21, trace22, trace23, 
       trace30, trace36,
       trace41, trace42, trace44,
       trace53, trace56, trace58,
       trace61]

py.iplot(data, filename='line-mode')

# relevant_words.index("continu")

# ind = [2, 3, 7, 10, 14,17,19,21, 22, 23, 24, 30, 36, 37, 38, 41, 42, 43, 44, 46, 53, 54, 56, 58,61]
# avg = []
# for i in ind:
#     avg.append((np.mean([y[0] for y in y_vals[i]]), i))

# sorted(avg)

# to change date - change the numbers and replace 'table' with the appropriate lines of code

table = relevant_data_df["WordCount"]
# relevant_data_df[relevant_data_df["Year"] <= 1990]["WordCount"]
# relevant_data_df[np.logical_and(relevant_data_df["Year"] <= 2000, relevant_data_df["Year"] > 1990)]["WordCount"]

word_count_all = [x[0].toarray()[0] for x in list(table)]

total_words = sum([sum(x) for x in word_count_all])

i = 0
new_word_count_all = []
while i < m.shape[1]:
    num_words = 0
    for arr in word_count_all:
        num_words += arr[i]
    new_word_count_all.append(num_words) 
    i += 1

word_freq = np.array(new_word_count_all) * 1000 / total_words
words = input_vectorizer_counts.get_feature_names()

words_and_word_freq = [x for x in zip(word_freq, words)]
words_and_word_freq = sorted(words_and_word_freq)
words_and_word_freq.reverse()

words_and_word_freq[0:50]

# with weighting

wtable = relevant_data_df["WeightedWordCount"]
# relevant_data_df[relevant_data_df["Year"] <= 1990]["WeightedWordCount"]
# relevant_data_df[np.logical_and(relevant_data_df["Year"] <= 2000, relevant_data_df["Year"] > 1990)]["WeightedWordCount"]

wword_count_all = [x[0].toarray()[0] for x in list(wtable)]

wtotal_words = sum([sum(x) for x in wword_count_all])

i = 0
new_wword_count_all = []
while i < m.shape[1]:
    num_words = 0
    for arr in wword_count_all:
        num_words += arr[i]
    new_wword_count_all.append(num_words) 
    i += 1

wword_freq = np.array(new_wword_count_all) * 1000 / wtotal_words
wwords = input_vectorizer.get_feature_names()

wwords_and_word_freq = [x for x in zip(wword_freq, wwords)]
wwords_and_word_freq = sorted(wwords_and_word_freq)
wwords_and_word_freq.reverse()

wwords_and_word_freq[0:50]

# graphs for reduction, increase, total words. t stats, stds under increase and reduction. 

# For each basis point change top words under both

# box and whisker plots

# tables for stemmer mappings, word counts during preprocessing

# most frequent terms across dates

# Doc filtering example appendix

# # for each document in train_df, run CountVectorizer to remove stop words.
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# import chardet
# import PyPDF2

# def input_documents(filenames):
#     text = ""
#     for filename in filenames:
#         with open(filename, 'rb') as input_file:
#             pdfReader = PyPDF2.PdfFileReader(input_file)
#             num_pages = pdfReader.numPages #later refactor to words within a doc
#             for i in range(0, num_pages):
#                 pageObj = pdfReader.getPage(i)
#                 text += " " + pageObj.extractText()
#     return [text]

# i = 0
# new_col_feature_names = []
# new_col_tokens = []
# print("Number Rows", relevant_data_df.shape[0])
# while i < relevant_data_df.shape[0]:
#     try:
#         document_paths = []
#         document_paths += np.array(relevant_data_df["Transcripts"])[i]
# #         tfidf_transformer = TfidfTransformer()
#         input_vectorizer = CountVectorizer(input="content", stop_words="english")
#         docs = input_documents(document_paths)
# #         tfidf_docs = tfidf_transformer.fit_transform(docs)
# #         print(tfidf_docs)
#         x = input_vectorizer.fit_transform(docs)
#         new_col_feature_names.append(list(zip(range(0,len(input_vectorizer.get_feature_names())),input_vectorizer.get_feature_names())))
#         new_col_tokens.append(x) 
#         print("Row " + str(i + 1) +  " Complete")
#         i += 1
#     except:
#         new_col_feature_names.append([None])
#         new_col_tokens.append(None)
#         print("Row " + str(i + 1) +  " Complete")
#         i += 1

# relevant_data_df["TokenCount"] = new_col_tokens
# relevant_data_df["FeatureNames"] = new_col_feature_names

