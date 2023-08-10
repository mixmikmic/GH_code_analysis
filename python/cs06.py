import pandas as pd
import numpy as np
from plotnine import *
import plotnine as pln

from IPython.display import Markdown, display

import warnings
warnings.filterwarnings('ignore')

# Load initial data
scriptures = pd.read_csv("scriptures.csv")
names = pd.read_csv("savior_names.csv")

# Isolate BOM data
bom = scriptures.loc[scriptures.volume_title == "Book of Mormon", ['book_title', 'scripture_text']]

# Group the 14 books together with all verses together
bom_books = bom.groupby('book_title')['scripture_text'].apply(lambda x: " ".join(x))
bom = pd.DataFrame(bom_books)
bom.reset_index(level=0, inplace=True)

# Replace the names of the Saviour with an ID
for name in names['name']:
    bom['scripture_text'] = bom['scripture_text'].str.replace(name, "<<name>>")

# Split on each name location
bom['scripture_text'] = bom['scripture_text'].str.split("<<name>>")

# Put each gap between references to the Saviour
# on its own row
gaps = bom.apply(lambda x: pd.Series(x['scripture_text']),axis=1).stack().reset_index(level=1, drop=True)
gaps.name = 'gaps'
bom = bom.drop('scripture_text', axis=1).join(gaps).reset_index(level=0)

# Calculte the average length
bom['length'] = bom['gaps'].str.split(" ").str.len()
average_lengths = bom.groupby('book_title', as_index=False).mean().sort_values('length').drop('index', axis=1)

avg = average_lengths['length'].mean()
string = 'The avergae number of words between mentions of the Saviour\'s name across the entire Book of Mormon is {:.2f}'.format(avg)
display(Markdown(string))

books_in_order = ['Helaman',
 'Alma',
 'Jarom',
 'Ether',
 'Mosiah',
 'Omni',
 '3 Nephi',
 'Jacob',
 '1 Nephi',
 'Words of Mormon',
 '2 Nephi',
 'Mormon',
 'Enos',
 '4 Nephi',
 'Moroni']

(ggplot(average_lengths, aes(x='book_title',y='length')) +
 geom_point() + 
 scale_x_discrete(limits = books_in_order) + 
 labs(title='Average Distance Between References to Christ',
      x='Book',
      y='Average Distance (number of words)') +
 theme_bw() +
 theme(axis_text_x=element_text(angle=45, hjust=1, vjust=1)))

