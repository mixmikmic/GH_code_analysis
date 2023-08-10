get_ipython().magic('matplotlib inline')

from word_count import WordCounter
from word_count import LetterCounter
from word_count import frequency_plot


STRING_MANUF_MULTI = 'This is?\r my |file.\nIt right\t 123 I suppose...\nThis is !really! test.\nI hope it, works'
STRING_MANUF_ONE = 'This is just another string but long and no newlines to test the read_in_string method. is is.'

TEXT_FRANKENSTEIN = './static/pg83.txt'
TEXT_MOON = './static/pg84.txt'

# Read-in the Frankenstein text file and return the 10 most common words and their counts.
# Passing length=None* will return all valid English words and their counts.
# *All words must be checked against the 230K Words dictionary so this can take a couple of minutes

stein = WordCounter().read_in_file(TEXT_FRANKENSTEIN, length=15)
stein

# Read-in a string and return the most common letters and their counts.
# Passing a length > than the actual number of letters in the counted text simply returns what was found.

manu = LetterCounter().read_in_string(STRING_MANUF_MULTI, length=100)
manu

frequency_plot(stein)

# Plot letter frequency

frequency_plot(manu)



