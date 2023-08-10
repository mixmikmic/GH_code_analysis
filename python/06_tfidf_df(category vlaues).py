cat_df = pd.read_csv('../resource/preprocess_df2.csv').ix[:,'director':'nation']
cat_df.head()

df = pd.DataFrame(columns=['string'])
for index, row in cat_df.iterrows():
    a = row['director'] +row['actors']+ str(row['film_rate'])+row['genre']+row['nation']
    #print(a)
    df.loc[len(df)] = a
df.head()

cat = np.array(df['string'])
cat

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
counts = count.fit_transform(cat)

print(counts[0])

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer().fit_transform(counts)
print(tfidf[0])

tfidf_arr = tfidf.toarray()
tfidf_arr.shape

tfidf_df = pd.DataFrame(tfidf_arr)
tfidf_df.head()

tfidf_df.to_csv('../resource/preprocess_tfidf_df.csv', index=False)

