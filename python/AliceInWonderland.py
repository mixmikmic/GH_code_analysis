import conx as cx

INPUT_FILE = "alice_in_wonderland.txt"

cx.download("http://www.gutenberg.org/files/11/11-0.txt", filename=INPUT_FILE)

# extract the input as a stream of characters
lines = []
with open(INPUT_FILE, 'rb') as fp:
    for line in fp:
        line = line.strip().lower()
        line = line.decode("ascii", "ignore")
        if len(line) == 0:
            continue
        lines.append(line)
text = " ".join(lines)
lines = None # clean up memory

chars = set([c for c in text])
nb_chars = len(chars)
char2index = dict((c, i) for i, c in enumerate(chars))
index2char = dict((i, c) for i, c in enumerate(chars))

nb_chars

char2index["a"]

index2char[5]

SEQLEN = 10
data = []
for i in range(0, len(text) - SEQLEN):
    inputs = [cx.onehot(char2index[char], nb_chars + 1) for char in text[i:i + SEQLEN]]
    targets = [cx.onehot(char2index[char], nb_chars + 1) for char in text[i + SEQLEN]][0]
    data.append([inputs, targets])
text = None # clean up memory

dataset = cx.Dataset()
dataset.load(data)
data = None # clean up memory; not needed

len(dataset)

cx.shape(dataset.inputs[0])

def onehot_to_char(vector):
    index = cx.argmax(vector)
    return index2char[index]

for i in range(10):
    print("".join([onehot_to_char(v) for v in dataset.inputs[i]]), 
          "->",
          onehot_to_char(dataset.targets[i]))

network = cx.Network("Alice in Wonderland")
network.add(
    cx.Layer("input", (SEQLEN, nb_chars + 1)),
    cx.SimpleRNNLayer("rnn", 128, 
                      return_sequences=False,
                      unroll=True),
    cx.Layer("output", nb_chars + 1, activation="softmax"),
)
network.connect()
network.compile(error="categorical_crossentropy", optimizer="rmsprop")

network.set_dataset(dataset)

network.summary()

network.dashboard()

def generate_text(sequence, count):
    for i in range(count):
        output = network.propagate(sequence)
        char = index2char[cx.argmax(output)]
        print(char, end="")
        sequence = sequence[1:] + [output]
    print()

for iteration in range(25):
    print("=" * 50)
    print("Iteration #: %d" % (network.epoch_count))
    results = network.train(1, batch_size=128, plot=False, verbose=0)   
    sequence = network.dataset.inputs[cx.choice(len(network.dataset))]
    print("Generating from seed: %s" % ("".join([onehot_to_char(v) for v in sequence])))
    generate_text(sequence, 100)
network.plot_results()

