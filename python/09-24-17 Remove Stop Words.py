# Load library
from nltk.corpus import stopwords

# You will have to download the set of stop words the first time
import nltk
nltk.download('stopwords')

# Create Word Tokens
tokenized_words = ['i', 'am', 'going', 'to', 'go', 'to', 'the', 'store', 'and', 'park']

# Load Stop Words
stop_words = stopwords.words('english')

# Show stop words
stop_words[:5]

# Remove stop words
[word for word in tokenized_words if word not in stop_words]

