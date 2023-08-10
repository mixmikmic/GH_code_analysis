import nltk
import pprint

tokenizer = None
tagger = None
def init_nltk():
    global tokenizer
    global tagger
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+|[^\w\s]+')
    tagger = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents())

def tag(text):
    tokenized = tokenizer.tokenize(text)
    tagged = tagger.tag(tokenized)
    tagged.sort(lambda x,y:cmp(x[1],y[1]))
    return tagged

text = """Mr. Blobby is a fictional character who featured on Noel
    Edmonds' Saturday night entertainment show Noel's House Party,
    which was often a ratings winner in the 1990s. Mr Blobby also
    appeared on the Jamie Rose show of 1997. He was designed as an
    outrageously over the top parody of a one-dimensional, mute novelty
    character, which ironically made him distinctive, absurd and popular.
    He was a large pink humanoid, covered with yellow spots, sporting a
    permanent toothy grin and jiggling eyes. He communicated by saying
    the word "blobby" in an electronically-altered voice, expressing
    his moods through tone of voice and repetition.

    There was a Mrs. Blobby, seen briefly in the video, and sold as a
    doll.

    However Mr Blobby actually started out as part of the 'Gotcha'
    feature during the show's second series (originally called 'Gotcha
    Oscars' until the threat of legal action from the Academy of Motion
    Picture Arts and Sciences[citation needed]), in which celebrities
    were caught out in a Candid Camera style prank. Celebrities such as
    dancer Wayne Sleep and rugby union player Will Carling would be
    enticed to take part in a fictitious children's programme based around
    their profession. Mr Blobby would clumsily take part in the activity,
    knocking over the set, causing mayhem and saying "blobby blobby
    blobby", until finally when the prank was revealed, the Blobby
    costume would be opened - revealing Noel inside. This was all the more
    surprising for the "victim" as during rehearsals Blobby would be
    played by an actor wearing only the arms and legs of the costume and
    speaking in a normal manner.[citation needed]"""

init_nltk()
tagged = tag(text)    
tag_list = list(set(tagged))
tag_list.sort(lambda x,y:cmp(x[1],y[1]))
pprint.pprint(tag_list)

tags = {}
for tag in tag_list:
    if tag[1] not in tags.keys():
        tags[tag[1]] = nltk.help.upenn_tagset(tag[1])

# Takes a little while too loads
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

# Let's print our sentences
print('\n-----\n'.join(sent_detector.tokenize(text.strip())))

nltk.download()

words = nltk.corpus.gutenberg.words(fileids='austen-emma.txt')
print words
print len(words)



