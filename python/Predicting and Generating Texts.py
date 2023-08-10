import conx as cx

def encode(s):
    """Convert string or char into integers"""
    if len(s) == 1:
        return (1 + ord(s.lower()) - ord('a')) if s.isalpha() else 0
    else:
        return cleanup([encode(c) for c in s])

def cleanup(items):
    """Remove repeated zeros"""
    retval = []
    for i in items:
        if ((i != 0) or 
            (len(retval) == 0) or 
            (retval[-1] != 0)):
            retval.append(i)
    return retval

def decode(n):
    """Convert integers into characters"""
    if isinstance(n, (list, tuple)):
        return [decode(v) for v in n]
    elif n == 0:
        return ' '
    else:
        return chr(ord('a') + int(n) - 1)

encode("H")

encode("Hello, world!")

encode("AaaA")

decode(8)

decode(encode("   what's     up  doc?   "))

"".join(decode(encode("   what's     up  doc?   ")))

corpus = """Four score and seven years ago our fathers brought forth on this continent, 
a new nation, conceived in Liberty, and dedicated to the proposition that all men are 
created equal. Now we are engaged in a great civil war, testing whether that nation, or 
any nation so conceived and so dedicated, can long endure. We are met on a great battle-field 
of that war. We have come to dedicate a portion of that field, as a final resting place 
for those who here gave their lives that that nation might live. It is altogether fitting 
and proper that we should do this. But, in a larger sense, we can not dedicate — we can not 
consecrate — we can not hallow — this ground. The brave men, living and dead, who struggled 
here, have consecrated it, far above our poor power to add or detract. The world will little 
note, nor long remember what we say here, but it can never forget what they did here. It is 
for us the living, rather, to be dedicated here to the unfinished work which they who fought 
here have thus far so nobly advanced. It is rather for us to be here dedicated to the great 
task remaining before us — that from these honored dead we take increased devotion to that 
cause for which they gave the last full measure of devotion — that we here highly resolve that 
these dead shall not have died in vain — that this nation, under God, shall have a new birth of 
freedom — and that government of the people, by the people, for the people, shall not perish 
from the earth."""

"".join(decode(encode(corpus)))

len_vocab = max(encode(corpus)) + 1
len_vocab

dataset = []
encoded_corpus = encode(corpus)
for i in range(len(encoded_corpus) - 1):
    code = encoded_corpus[i]
    next_code = encoded_corpus[i + 1]
    dataset.append([[code], cx.onehot(next_code, len_vocab)])

net = cx.Network("Given 1 - Predict 1")
net.add(cx.Layer("input", 1), 
        cx.EmbeddingLayer("embed", 26, 64),  # in, out
        cx.FlattenLayer("flatten"),
        cx.Layer("output", 26, activation="softmax"))
net.connect()
net.compile(error="categorical_crossentropy", optimizer="adam")

net.dataset.load(dataset)

net.dashboard()

if net.saved():
    net.load()
    net.plot_results()
else:
    net.train(30, accuracy=.95, save=True)

def generate(net, count, len_vocab):
    retval = ""
    # start at a random point:
    inputs = cx.choice(net.dataset.inputs)
    # now we get the next, and the next, ...
    for i in range(count):
        # use the outputs as a prob distrbution
        outputs = net.propagate(inputs)
        code = cx.choice(p=outputs)
        c = decode(code)
        print(c, end="")
        retval += c
    return retval

generate(net, 500, len_vocab)

net2 = cx.Network("Given 5 - Predict 1")
net2.add(cx.Layer("input", 5),
         cx.EmbeddingLayer("embed", 26, 64),
         cx.FlattenLayer("flatten"),
         cx.Layer("output", 26, activation="softmax"))
net2.connect()
net2.compile(error="categorical_crossentropy", optimizer="adam")

dataset = []
encoded_corpus = encode(corpus)
for i in range(len(encoded_corpus) - 5):
    code = encoded_corpus[i:i+5]
    next_code = encoded_corpus[i + 5]
    if len(code) == 5:
        dataset.append([code, cx.onehot(next_code, len_vocab)])

net2.dataset.load(dataset)

for i in range(10):
    print(i, decode(net2.dataset.inputs[i]), decode(cx.argmax(net2.dataset.targets[i])))

net2.dashboard()

if net2.saved():
    net2.load()
    net2.plot_results()
else:
    net2.train(80, accuracy=.95, plot=True, save=True)

def generate2(net, count, len_vocab):
    # start at a random point:
    inputs = cx.choice(net.dataset.inputs)
    retval = "".join(decode(inputs))
    print(retval, end="")
    # now we get the next, and the next, ...
    for i in range(count):
        # use the outputs as a prob distrbution
        outputs = net.propagate(inputs)
        pickone = cx.choice(p=outputs)
        inputs = inputs[1:] + [pickone]
        c = decode(pickone)
        print(c, end="")
        retval += c
    return retval

generate2(net2, 1000, 26)

net3 = cx.Network("LSTM - Many to One")
net3.add(cx.Layer("input", 40), # sequence length
         cx.EmbeddingLayer("embed", 26, 64), # sequence_length from input
         cx.LSTMLayer("lstm", 64),
         cx.Layer("output", 26, activation="softmax"))
net3.connect()
net3.compile(loss='categorical_crossentropy', optimizer='adam')

dataset = []
encoded_corpus = encode(corpus)
for i in range(len(encoded_corpus) - 40):
    code = encoded_corpus[i:i+40]
    next_code = encoded_corpus[i + 40]
    if len(code) == 40:
        dataset.append([code, cx.onehot(next_code, len_vocab)])

net3.dataset.load(dataset)

dash = net3.dashboard()
dash

dash.propagate(net3.dataset.inputs[0])

if net3.saved():
    net.load()
    net.plot_results()
else:
    net3.train(150, save=True)

def generate3(net, count, len_vocab):
    # start with a full sentence:
    inputs = cx.choice(net.dataset.inputs)
    print("".join(decode(inputs)), end="")
    for i in range(count):
        outputs = net.propagate(inputs)
        pickone = cx.choice(p=outputs)
        inputs = inputs[1:] + [pickone]
        print(decode(pickone), end="")

generate3(net3, 500, len_vocab)

net4 = cx.Network("Many-to-Many LSTM")
net4.add(cx.Layer("input", None),  # None for variable number 
         cx.EmbeddingLayer("embed", 26, 64),
         cx.LSTMLayer("lstm", 256, return_sequences=True), # , stateful=True
         cx.Layer("output", 26, activation='softmax', time_distributed=True))
net4.connect()
net4.compile(error="categorical_crossentropy", optimizer="adam")
net4.model.summary()

dataset = []
encoded_corpus = ([0] * 39) + encode(corpus)
for i in range(len(encoded_corpus) - 40):
    code = encoded_corpus[i:i+40]
    next_code = encoded_corpus[i+1:i+40+1]
    if len(code) == 40:
        dataset.append([code, list(map(lambda n: cx.onehot(n, len_vocab), next_code))])

cx.shape(dataset[0][1])

net4.dataset.load(dataset)

dash4 = net4.dashboard()
dash4

dash4.propagate([13])

dash4.propagate([13, 21])

net4.train(10)

def generate4(net, count, len_vocab):
    letters = [cx.choice(len_vocab)] # choose a random letter
    for i in range(count):
        print(decode(letters[-1]), end="")
        outputs = net.propagate(letters)
        if len(cx.shape(outputs)) == 1:
            p = outputs
        else:
            p = outputs[-1]
        letters.append(cx.choice(p=p))
        letters = letters[-40:]

generate4(net4, 500, len_vocab)

net4.picture([1, 100, 4, 2])

net4.picture([3, 69, 200, 10, 4])

output = net4.propagate(range(4))

cx.shape(net4.dataset.inputs[43:47])



