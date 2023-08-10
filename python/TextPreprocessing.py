import pprint, string, nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

cv = CountVectorizer(analyzer = "word", lowercase = True)

cv

myText = "Hello! My name is Jacky Zhao. I'm an aspiring data scientist. Follow me on twitter @iamdatabear. The is my text analysis practice!"

myText

cv1 = CountVectorizer(lowercase = True)
cv2 = CountVectorizer(stop_words = "english", lowercase = True)

tk_func1 = cv1.build_analyzer()
tk_func2 = cv2.build_analyzer()

pp = pprint.PrettyPrinter(indent = 2, depth = 1, width = 80, compact = True)

example1 = tk_func1(myText)
print("Tokenization for 1: \n", example1)

example2 = tk_func2(myText)
print("Tokenization for 2: \n", example2)

example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]

stemmer = PorterStemmer()

for w in example_words:
    print(stemmer.stem(w))

newText = "It is important to be very pythonly while you are pythoning with pythong. All pythoners have pythoned poorly at least once!"

newText

tokens = nltk.word_tokenize(newText)

print(tokens)

tokens = [token for token in tokens if token not in string.punctuation]

print(tokens)

for w in tokens:
    print(stemmer.stem(w))

lemmatizer = WordNetLemmatizer()

lemmatizer.lemmatize("dogs")

words = ["going", "gone", "goes", "went"]

print("Stemming: \n")
for w in words:
    print(w,"becomes", stemmer.stem(w))

print("Lemmatize without context:\n")
for w in words:
    print(w, "becomes", lemmatizer.lemmatize(w))

print("Lemmatize WITH context: \n")
for w in words:
    print(w, "becomes", lemmatizer.lemmatize(w, pos = "v"))

s = "This is a simple sentence. Let's SeE IF iT cAn fiGuRe tHiS cRaZy sEnTenCe ouT!"
s

tokens = word_tokenize(s)

print(tokens)

tokens_pos = pos_tag(tokens)

print(tokens_pos)

word_and_pos = {}

for tp in tokens_pos:
    word_and_pos[tp[0]] = tp[1]

len(word_and_pos)

print(word_and_pos)

stringAction = "We are meeting"
stringNoun = "We had a meeting"

cv = CountVectorizer(stop_words = "english", lowercase = True)
tk_function = cv.build_analyzer()

pp = pprint.PrettyPrinter(indent = 2, depth = 1, width = 80, compact = True)

print("Tokenizations:\n")
print("'{}':".format(stringAction))
pp.pprint(tk_function(stringAction))

print("Tokenization:\n")
print("'{}':".format(stringNoun))
pp.pprint(tk_function(stringNoun))

stemmer = PorterStemmer()

print(stemmer.stem(stringAction))

print(stemmer.stem(stringNoun))

lem = WordNetLemmatizer()

stringActionTokens = nltk.word_tokenize(stringAction)
sat = [t for t in stringActionTokens if t not in string.punctuation]
print(sat)

stringNounTokens = nltk.word_tokenize(stringNoun)
snt = [t for t in stringNounTokens if t not in string.punctuation]
print(snt)

for w in sat:
    print(lem.lemmatize(w))

for w in snt:
    print(lem.lemmatize(w))

print(pos_tag(stringActionTokens))

print(pos_tag(stringNounTokens))

