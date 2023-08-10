import pandas as pd

data = pd.read_csv('writer_categories.csv', names = ['name', 'categories'])

data.head()

data.describe()

print len(data[data.categories.isnull()])

print len(data[data.categories.notnull()])
have_cats = data[data.categories.notnull()]
print len(have_cats)

no_categories = data[data.categories.isnull()]['name'].tolist()
print no_categories
#checked the first before ocmp dies - imdb had gende for one out of two of missing

#Get a list of ALL categories
cats = data.categories
all_cats = []
for cat in cats:
    split= str(cat).split(',')
    for sub_cat in split:
        replacables = ["\\'Category:", "Category:", '"', "u'", "'", '[[', ']]', '[', ']']
        for repl in replacables:
            sub_cat = sub_cat.replace(repl, "")
        all_cats.append(sub_cat)

print all_cats[:10]

unique = list(set(all_cats))
print unique

for x in unique:
    if 'women' in x:
        print x

for x in unique:
    if 'writers' in x:
        print x

for x in unique:
    if 'directors' in x:
       print x

women = have_cats[have_cats['categories'].str.contains("women")]
print len(women)

queer = have_cats[have_cats['categories'].str.contains("LGBT")]
print len(queer)



