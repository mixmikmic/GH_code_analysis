import pandas as pd

talk = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/enwiki/talk_filtered_3.csv')
talk

# Sample text files for e-thos

for i,t in talk['text'][:100].iteritems():
    with open('/home/michael/school/research/e-thos/E-thos_Python_4/corpus/wp_talk{}.txt'.format(i), 'w') as out:
        out.write(t)

