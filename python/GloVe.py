directory = 'brown-europarl'
filename = 'brown-europarl'
is_text_already_split_on_sentences = True
with open('corpora/{}/{}.txt'.format(directory, filename), 'r') as fd:
    text = fd.read()

do_break_context_window_on_sentences = False

import re
# Remove punctuation, special characters.
# TODO: !!! Keep apostrophe (') when in the middle of a word.
text = re.sub('[\',-=;:"#+<>%$_()&*@^\[\]`~{}|\\\\]+', ' ', text)
# Create sentences, or boundaries for our context window.
if do_break_context_window_on_sentences:
    if is_text_already_split_on_sentences:
        # The text is split on sentence endings.
        text = re.sub('[.?!]+', ' ', text)
    else:
        # Create sentence endings in a naive way.
        text = re.sub('[.?!]+', '\n', text)
    sentences = text.split('\n')
else:
    # Treat the entire text as a single sentence.
    text = re.sub('[.?!]+', ' ', text)
    sentences = [text]
# TODO: Do we have to convert all of the text to lowercase?
token_sentences = [[token.lower() for token in sentence.strip().split()] for sentence in sentences]
print("Sentences: {}".format(len(token_sentences)))
token_count = len(token_sentences[0])
vocab = set(token_sentences[0])
for sentence in token_sentences[1:]:
    token_count += len(sentence)
    vocab.update(sentence)
print("Tokens: {}".format(token_count))
vocab = list(vocab)
vocab.sort()
m = len(vocab)
print("Vocabulary terms: {}".format(m))

with open('corpora/{}/{}.vocab'.format(directory, filename), 'w') as fd_vocab:
    for i in range(m):
        fd_vocab.write("{}\n".format(vocab[i]))

lookup = {}
for i, v in enumerate(vocab):
    lookup[v] = i
lookup['the']

import numpy as np
# r = Window radius: Terms to the left or right of a
#  given term defined as being in the same "context".
r = 8
co = np.zeros((m, m))
def occur(co, token, other):
    """
    Update the co-occurrence matrix when a word appears in another word's context.
    """
    i = lookup[token]
    j = lookup[other]
    co[i, j] += 1

for sentence in token_sentences:
    for t, token in enumerate(sentence):
        # Count co-occurrences to the left of this term.
        for other in sentence[max(0, t - r):t]:
            occur(co, token, other)
        # Count co-occurrences to the right of this term.
        for other in sentence[t + 1:min(t + 1 + r, len(sentence))]:
            occur(co, token, other)

vector_dim = 200
iterations = 25
co_max = np.max(co)
weight_alpha = 3/4
learning_rate = .01

center_embedding = np.random.uniform(low=-1, high=1, size=(m, vector_dim))
context_embedding = np.random.uniform(low=-1, high=1, size=(m, vector_dim))

center_bias = np.random.uniform(low=-1, high=1, size=(m))
context_bias = np.random.uniform(low=-1, high=1, size=(m))

##### Variable update historical arrays
center_history = np.zeros((m, vector_dim)) + .1
context_history = np.zeros((m, vector_dim)) + .1
bias_center_history = np.zeros(m) + .1
bias_context_history = np.zeros(m) + .1

def weight_fun(x, co_max, alpha):
    if x >= co_max:
        return 1
    return np.power(x/co_max, alpha)

losses = []
for iters in range(iterations):
    global_loss = 0
    for i in range(m):
        for j in range(m):
            count = co[i,j]
            if count == 0:
                continue
            center = center_embedding[i,:]
            context = context_embedding[j,:]
            b1 = center_bias[i]
            b2 = context_bias[j]
            weight = weight_fun(count, co_max, weight_alpha)
            inner_loss = np.dot(center, context) + b1 + b2 - np.log(count)
            loss = weight * np.square(inner_loss)
            global_loss += loss

            ### Compute Gradients
            grad_center = weight * inner_loss * context
            grad_context = weight * inner_loss * center
            grad_bias_center = weight * inner_loss
            grad_bias_context = weight * inner_loss

            center_embedding[i,:] -=  learning_rate * (grad_center  / np.sqrt(center_history[i,:]))
            context_embedding[j,:] -= learning_rate * (grad_context / np.sqrt(context_history[j,:]))
            center_bias[i] -=  learning_rate * (grad_bias_center / np.sqrt(bias_center_history[i]))
            context_bias[j] -= learning_rate * (grad_bias_context / np.sqrt(bias_context_history[j]))

            center_history[i,:] += np.square(grad_center)
            context_history[j,:] += np.square(grad_context)
            bias_center_history[i] += np.square(grad_bias_center)
            bias_context_history[j] += np.square(grad_bias_context)
    losses.append(global_loss)
    print("Completed iteration: {}".format(iters))

# import matplotlib.pyplot as plt
# plt.plot(losses)
# plt.xlabel("Iterations")
# plt.ylabel("Loss")
# plt.show()

with open("corpora/{}/{}_{}_{}_{}_skip.center".format(directory, filename, r, vector_dim, iterations), "w") as fd_center:
    for i in range(m):
        for j in range(vector_dim):
            fd_center.write("{} ".format(center_embedding[i][j]))
        # Write the bias weight.
        fd_center.write("{}\n".format(center_bias[i]))

with open("corpora/{}/{}_{}_{}_{}_skip.context".format(directory, filename, r, vector_dim, iterations), "w") as fd_context:
    for i in range(m):
        for j in range(vector_dim):
            fd_context.write("{} ".format(context_embedding[i][j]))
        # Write the bias weight.
        fd_context.write("{}\n".format(context_bias[i]))



