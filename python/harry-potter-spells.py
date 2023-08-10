import pandas as pd
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

# Reading in the list of books
books = glob.glob(r'.\books\*.txt')

# Putting the books into a dataframe
contents = [open(book,encoding='Latin5').read() for book in books]
books_df = pd.DataFrame({
    'book': books,
    'body': contents,
})
books_df

# Reading in the list of spells
spells = pd.read_csv('spells.csv')['spell'].tolist()

# This counts ALL the words, relative numbers
#vectorizer = TfidfVectorizer(use_idf=False, norm='l1',stop_words='english')
#matrix = vectorizer.fit_transform(books_df['body'])
#matrix

# This counts ALL the words, absolute numbers
vectorizer = CountVectorizer(stop_words='english')
matrix = vectorizer.fit_transform(books_df['body'])
matrix

# And this creates a grid out of it.
# NOTE: the (number)harry words came from page enumeration strings.
results = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names())
results.head()

# But I want to look ONLY at the spells.
# This recquires some handling, though. I want to look just at the spells that appear in the books - there are some more in the list.
# Also, if a spell is composed of more than two words, I'll look just at the first one. e.g 'Avada Kedavra' will be AVADA.
spells = [spell.split()[0] for spell in spells]

# Make the list into a dataframe so I can filter only the spells that appear in the book
spells_df = pd.DataFrame({
    'spell': spells,
})
spells_df.head()

# Keeping only the spells that appear in the book - and in the books_df headers, this way
spells_df = spells_df[spells_df['spell'].str.lower().isin(list(results.columns.values))]

# Making everything lowercase in order to compare
spells_df['spell'] = spells_df['spell'].str.lower()

# And getting only the magic words!
results_spells = results[spells_df['spell'].tolist()]

# What is the total % of magic words
results_spells['magic_%'] = results_spells.sum(axis=1)

# What is the most spell-intenstive book?
results_spells['magic_%'].sort_values(ascending=False)

# Let's plot a line chart and see the trend.
ax = results_spells['magic_%'].plot()
ax.set_title("Evolution of magic usage")
plt.savefig('magic-usage.png')

# FUN FACT: As Avada Kedavra is the wizarding-world equivalent of a gunshot, we can say that Harry Potter and The Golbet of Fire was the most violent book in the series, proportionally.
results_spells['avada'].plot()

# And it was also the most SADISTIC one, with abuse of the torture spell.
results_spells['crucio'].plot()

# And as the series progressed, people got lazy - 'accio' is a spell that finds and brings things to the hands of the wizard.
results_spells['accio'].plot()

results_spells

# Making index match book order
results_spells.index = [1,2,3,4,5,6,7]

# So let's plot ALL THE SPELLS
# SIZE IN PIXELS: 470 -168
sns.set_style("white")
for spell in spells_df.spell.unique():
    fig, ax = plt.subplots(figsize=(6,2))
    results_spells[spell].plot(ax=ax,color='green')
    sns.despine(left=True,bottom=True,trim=True)
    plt.savefig((r"C:\Users\Avell\Desktop\the-lede-program\data-studio\harry-potter-spells\visuals\spells-lines\\" + spell + '-count.svg'), transparent=True)

    



