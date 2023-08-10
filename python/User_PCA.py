import pandas as pd
from collections import Counter

df = pd.read_csv('/Users/wallace/Documents/GitHub/Capstone/Android_Malware_Capstone/data/Volunteer_Survey.csv')

df2 = df.copy()
cols = ['{:03d}'.format(i) for i in xrange(df.shape[1])]
df2.rename(columns=dict(zip(df.columns,cols)), inplace=True)
for num in ['003', '065']:
    df2[num] = df2[num].map({'no':0, 'yes':1})
for num in ['007', '008', '048', '081', '085', '086', '087','088']:   
    df2[num] = df2[num].map({'No':0, 'Yes':1})
df2['009'] = df2['009'].map({'Male':1, 'Female':0})
df2['010'] = df2['010'].map({'Bachelors degree': 3,
         'Elementry School graduate': 1,
         'High school graduate': 2,
         "Master's degree": 4,
         'Other': 5})
df2['011'] = df2['011'].map({'English': 1, 'Hebrew': 0})
df2['012'] = df2['012'].map({'English': 0,
         'French': 1,
         'Hebrew': 2,
         'None': 3,
         'Other': 4,
         'Russian': 5})
df2['014'] = df2['014'].map({'Married':1, 'Single':0})
df2['016'] = df2['016'].map(lambda x: int(x) if x != 'more' else int(8))
df2['017'] = df2['017'].map({'Apple (Mac)': 0,
         'PC (Windows)': 1,
         '\xc3\x97\xc5\x93\xc3\x97\xc2\x90 \xc3\x97\xc5\xbe\xc3\x97\xc2\xa9\xc3\x97 \xc3\x97\xe2\x80\x9d': 1})
df2['022'] = df2['022'].map({'2-4 times a day': 4,
         '5-8 times a day': 8,
         '9-15 times a day': 15,
         'more than 16 times a day': 20})
df2['035'] = df2['035'].map({'for a minute': 1,
         'for an hour': 60,
         'for several minutes': 15,
         'less than a minute': 0})
df2['047'] = df2['047'].map({'ambidextrous': 0, 'left handed': -1, 'right handed': 1})
df2['049'] = df2['049'].map({"I don't drive": 0,
         'In a smartphone holder (held for the driver to see)': 5,
         'In my bag / purse': 4,
         'In my pocket': 3,
         'On a seat or other surface': 2,
         'rarely drive.': 1})
for num in ['{:03d}'.format(i) for i in xrange(50, 55)]:
    df2[num] = df2[num].map({'1-5 times a month': 36,
         '1-5 times a week': 156,
         '6-10 times a week': 416,
         'Never': 0,
         'a few times a year': 3,
         'more than 10 times a week': 624})
for num in ['{:03d}'.format(i) for i in xrange(55, 60)]:
    df2[num] = df2[num].map({'1-2 times a month': 24,
         '1-3 times a year': 3,
         '3-6 times a month': 60,
         '4-8 times a year': 8,
         'Never': 0,
         'a few times every week': 156})
for num in ['{:03d}'.format(i) for i in xrange(60, 64)]:
    df2[num] = df2[num].map({'1 time': 1,
         '2-4 times': 4,
         '5-8 times': 8,
         '9 or more': 15,
         'Never': 0})
df2['078'] = df2['078'].map({'Accept the changes, if after carefully reviewing them they seem resonable': 1,
         'Accept the changes, no matter what (I just want to play that game)': 0,
         'Reject the changes no matter what, and stop the update': 2,
         'Reject the changes no matter what, and then uninstall the game': 3})
df2['064'] = df2['064'].map({'Days': 1, 'Months': 3, 'Not relevant': 0, 'Weeks': 2})
df2['081'] = df2['081'].map({0.0:0, 1.0:1, None:0})
df2['082'] = df2['082'].map({'Always have installed from official Markets': 1,
         "I don't understand the question.": 0,
         'I have on ocassion installed APKs manually': 2})
df2['089'] = df2['089'].map({'Fingerprint': 4,
         "I don't lock my phone (A swipe or button press unlocks my phone)": 0,
         'PIN': 2,
         'Password': 5,
         'Swipe finger pattern': 3})
for num in ['{:03d}'.format(i) for i in xrange(102, 106)]:
    df2[num] = df2[num].map({'Casually': 4,
         'Frequently': 5,
         'Never': 0,
         'When I am obligated': 1,
         "When I'm reminded": 2,
         'When something stops working': 3})
df2['116'] = df2['116'].map({'All of the time': 3, 'Often': 2, 'Rarely': 1, 'What is that?': 0})
# for num in ['007', '008', '081']:
#     df2[num] = df2[num].astype(int)
df2.drop(['004', '005', '006', '066', '067', '068', '069', '070',          '071', '072'], axis=1, inplace=True)
# df2.loc[:10,'021':'030']

# Counter(df2['002'])

# df2.info(verbose=True)

for item in zip(cols, df.columns):
    print item



from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(df2.loc[:,'023':'034'].dropna(axis=1, how='any'))

X.shape

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
model=TSNE(perplexity=10, method='exact', verbose=1)
t_fit = model.fit_transform(X)
# print t_fit.T
plt.scatter(t_fit.T[0], t_fit.T[1])
plt.show

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
p_fit = pca.fit_transform(X)
plt.scatter(p_fit.T[0], p_fit.T[1])
plt.show



