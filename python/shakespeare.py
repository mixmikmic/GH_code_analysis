from collections import namedtuple
from shakespeare import *
from torch.utils.data.dataset import TensorDataset
import torch
from IPython.display import display
from inferno.trainers.basic import Trainer

# Configure arguments
# Refer to shakespeare.py for training details
args = namedtuple('args',
                  [
                      'batch_size',
                      'save_directory',
                      'epochs', 
                      'cuda',
                      'batch_len',
                      'embedding_dim',
                      'hidden_dim'])(
    64,
    'output/shakespeare',
    20,
    False,
    200,
    128,
    256)

# Read and process data
corpus = read_corpus()
chars, charmap = get_charmap(corpus)
charcount = len(chars)

# What is the size and shape of our data?
print("Total character count: {}".format(len(corpus)))
print("Unique character count: {}".format(len(chars)))

# What does the text look like?
context = 256
print("{} ... {}".format(corpus[:context], corpus[-context:]))

# Break corpus into subsequences. 
# This is a simple and dirty method of making a dataset.
array = map_corpus(corpus, charmap)
targets = batchify(array, args=args)
inputs = make_inputs(targets)
dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))

# The network itself
class ShakespeareModel(nn.Module):
    def __init__(self, charcount, args):
        super(ShakespeareModel, self).__init__()
        self.charcount = charcount
        self.embedding = nn.Embedding(num_embeddings=charcount + 1, embedding_dim=args.embedding_dim)
        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_dim, batch_first=True),
            nn.LSTM(input_size=args.hidden_dim, hidden_size=args.hidden_dim, batch_first=True),
            nn.LSTM(input_size=args.hidden_dim, hidden_size=args.embedding_dim, batch_first=True)])
        self.projection = nn.Linear(in_features=args.embedding_dim, out_features=charcount)

    def forward(self, input, forward=0, stochastic=False):
        h = input  # (n, t)
        h = self.embedding(h)  # (n, t, c)
        states = []
        for rnn in self.rnns:
            h, state = rnn(h)
            states.append(state)
        h = self.projection(h)
        if stochastic:
            gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
            h += gumbel
        logits = h
        if forward > 0:
            outputs = []
            h = torch.max(logits[:, -1:, :], dim=2)[1] + 1
            for i in range(forward):
                h = self.embedding(h)
                for j, rnn in enumerate(self.rnns):
                    h, state = rnn(h, states[j])
                    states[j] = state
                h = self.projection(h)
                if stochastic:
                    gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
                    h += gumbel
                outputs.append(h)
                h = torch.max(h, dim=2)[1] + 1
            logits = torch.cat([logits] + outputs, dim=1)
        return logits


def generate(model, sequence_length, batch_size, args, stochastic=False, inp=None):
    if inp is None:
        inp = Variable(torch.zeros(batch_size, 1)).long()
        if args.cuda:
            inp = inp.cuda()
    model.eval()
    logits = model(inp, forward=sequence_length, stochastic=stochastic)
    classes = torch.max(logits, dim=2)[1]
    return classes

# Train or load a model
checkpoint_path = os.path.join(args.save_directory, 'checkpoint.pytorch')
if not os.path.exists(checkpoint_path):
    model = ShakespeareModel(charcount=charcount, args=args)
    train_model(model=model, dataset=dataset, args=args)
else:
    trainer = Trainer().load(from_directory=args.save_directory)
    model = ShakespeareModel(charcount=charcount, args=args)
    model.load_state_dict(trainer.model.state_dict())

# Generate deterministic text
generated = generate(model, sequence_length=2000, batch_size=2, stochastic=False, args=args).data.cpu().numpy()
text = to_text(preds=generated, charset=chars)
for i, t in enumerate(text):
    print("Deterministic #{}: {}".format(i,t))
# What do you think is going on here? Can you guess why the outputs are blank?

# Seed deterministic text
seeds = ['KING RICHARD', 'KING RICHARD', 'Enter Falsta', 'SHAKESPEARE ']
assert len(set(len(s) for s in seeds)) == 1
inp = np.array([[charmap[c] for c in l] for l in seeds], dtype=np.int64)
inp = np.pad(inp + 1, [(0, 0), (1, 0)], mode='constant')
inp = Variable(torch.from_numpy(inp))
if args.cuda:
    inp = inp.cuda()
# Generate stochastic text
generated = generate(model, sequence_length=2000, batch_size=5, stochastic=False, inp=inp,
                     args=args).data.cpu().numpy()
text = to_text(preds=generated, charset=chars)
for i, (s, t) in enumerate(zip(seeds, text)):
    print("Deterministic #{} (seed={}): {}".format(i, s, t))

# Generate stochastic text
generated = generate(model, sequence_length=2000, batch_size=5, stochastic=True, args=args).data.cpu().numpy()
text = to_text(preds=generated, charset=chars)
for i, t in enumerate(text):
    print("Stochastic #{}: {}".format(i,t))



