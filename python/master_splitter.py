import warnings
warnings.filterwarnings('ignore')

from bs4 import BeautifulSoup
from gensim import corpora, models
from gensim.models import ldamodel
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from requests import get
from stop_words import get_stop_words
from string import punctuation

stop_words = get_stop_words('english')

tokenizer = RegexpTokenizer(r'\w+')

porter_stemmer = PorterStemmer()

def clean_and_tokenize_text(text):
    raw = text.lower()
    tokens = tokenizer.tokenize(raw)
    stemmed = [porter_stemmer.stem(t) for t in tokens]
    cleaned = [t for t in stemmed if t not in stop_words]
    return cleaned

def parse_html_pages(urls):
    page_contents = []
    for url in urls:
        response = get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.body.text
        tokens = clean_and_tokenize_text(text)
        page_contents.append(tokens)
    return page_contents

urls = [
    "https://en.wikipedia.org/wiki/National_Basketball_Association",
    "https://en.wikipedia.org/wiki/Architecture",
    "https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss",
    "https://en.wikipedia.org/wiki/LeBron_James",
    "https://en.wikipedia.org/wiki/Number_theory",
    "https://en.wikipedia.org/wiki/Ronaldo_(Brazilian_footballer)",
    "https://en.wikipedia.org/wiki/Tennis",
    "https://en.wikipedia.org/wiki/Mathematical_logic",
    "https://en.wikipedia.org/wiki/Computer_science",
    "https://en.wikipedia.org/wiki/Ultimate_(sport)",
    "https://en.wikipedia.org/wiki/Leonardo_da_Vinci",
    "https://en.wikipedia.org/wiki/Michelangelo",
    "https://en.wikipedia.org/wiki/St._Peter%27s_Basilica"
]

true_labels = [0, 1, 2, 0, 2, 0, 0, 2, 2, 0, 1, 1, 1]

k = 2

texts = parse_html_pages(urls)
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda = ldamodel.LdaModel(corpus, num_topics=k, id2word=dictionary, passes=20)

results = [lda[dictionary.doc2bow(text)] for text in texts]
labels = [max(result, key=lambda tup: tup[1])[0] for result in results]

print 'predicted:', labels
print 'true lbls:', true_labels

