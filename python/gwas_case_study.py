from models.models import SequenceDNN_Regression
import numpy as np

model = SequenceDNN_Regression.load('models/models/145_weighted.arch.json', 'models/models/145_weighted.weights.h5')

# One hot encode DNA sequences the standard way.
bases = ['A', 'T', 'C', 'G']

def one_hot_encode_seq(seq):
    result = np.zeros((len(bases), len(seq)))

    for i, base in enumerate(seq):
        result[bases.index(base), i] = 1

    return result

def seqs_to_encoded_matrix(seqs):
    # Wrangle the data into a shape that Dragonn wants.
    result = np.concatenate(
        map(one_hot_encode_seq, seqs)
    ).reshape(
        len(seqs), 1, len(bases), len(seqs[0])
    )

    # Check we actually did the encoding right.
    for i in range(len(seqs)):
        for j in range(len(seqs[0])):
            assert sum(result[i, 0, :, j]) == 1

    return result

seq = 'ttttcacagattgagctgcttgaggacaagggtcatgtgttcatctttgGGTTGTACACCACCTTGAATGAATGAAAGGACAAAAGTCTTTATAGAGTCATCACTGCTGCCCATTTCTCTGCCTCACTTGGTTGTAGAAGCCAC'.upper()
seqs = [seq, seq[:72] + 'G' + seq[73:]]
print seqs[0][72], seqs[1][72]
X = seqs_to_encoded_matrix(seqs)
model.predict(X)

seq = 'tggagattcttttttccactgggtcagttctgctattaatagttgtgattgcattatgaaattcttatagtgtgtttttcagctctatcaaattggttatgtccttatctatactggctattttatctatcagctcctgcattg'.upper()
seqs = [seq, seq[:71] + 'A' + seq[72:]]
print seqs[0][71], seqs[1][71]
X = seqs_to_encoded_matrix(seqs)
model.predict(X)

seq = 'cagagtactttgtgcgcatgcttctgttaccacattaactcatatgatagttttcacagattgagctgcttgaggacaagggtcatgtgttcatctttgGGTTGTACACCACCTTGAATGAATGAAAGGACAAAAGTCTTTATA'.upper()
seqs = [seq, seq[:122] + 'G' + seq[123:]]
print seqs[0][122], seqs[1][122]
X = seqs_to_encoded_matrix(seqs)
model.predict(X) 

seqs = ['AGTGCAGTGGCGTGATCTCAGCTCACTGCATCCCCAACCTCCTGGGCTTGAGTGATCTTCCCACCTCAGCCTCCCGAGTAGCTGGGAACACAGGCACACACTACCATGCCTGGCTAATTCTTTGTATTTTTGGTAGAAGTGGGG', 'GCCACCATGCCCGGCTAATTTTTTTATTTTTAGTGGAGATGAGGTTTTACCATGTTGGCCAGGCTGGTCTCCAACTCCTGACCTCAGGTGATCCACCCACCTTGGCCACCCAAAGTGCTGGGATTACAGGCGTGAGCCACTGCA', 'AAGTGTCTAGTCAACTTAATTGAGAAGGTGGAATCCTCCTATCCCTGAACTCGGGGGAATGGAATCTCGCTGATCTTCCAGGACTAGCTCCCTGATCATTCCAGCCCCTCTGAACAACAGGGCCCCAGGAAAATCTCCAGGTCC', 'CCAAGATCACCCCATTGCACTCCAGCCTGGATAAAAAGAGTGAAACTCTGTCTCAAAAAAAAAAAAAAAGAACACCGAATCCCTGGCCAGGCACAGTGGCTCATACCTATAATCCCAGCACTTTGGGAGGCCAAGGAGGAAAGA', 'TTGTCAAAAATTGCAATTGTCATTCAATACACATGTTTGAGCACACAATGAGCTAACTTTTGGGAATTCAAAGATAAAAAATCATGCTGTCTGCCTTGCAGAGGGTGCACAAACCAGTGATGGAAACAGTATGGGGCACAGGAA', 'GCTTTTAATGTTGCAGCTCGGGGAGTTAAAGAAGGTCGTAATAGTTTATTTTCTTGGTTAGCTGAAATATGGATTAAAAGGTGGCCCACTGTGAGCAAGCTGGAAATGTCTGATCTCCCTTGGTTTAATGTAGAGGAAGGAATT', 'AAAGAGCCAGGATGACCATTTGGACCTGATTTTACTGGGAGGGGAGAGGGGCAAAGAAGGGAGTTGCTGTTCCCTAAAATGAGGAACCCCTCAGCCTTCGCATTTTCCTCTTGAGTCCCACAAAGGAGCAGCAACTTTACCCAC', 'AAGAAGAAGAGGGCTCCCTGCTTCTAGTGAGCAAAGGCAGTGCCTGAGCTTCTACAGCCCTTCGTATTTATTGGGTAACAAGAGCAAGGAGGAAGAGGTAATGATTGGTCAGCTGCTTAATTAATCACAGGTTCATATTATTAC', 'CTTCCTGCGGCGCAAGCTGCGCACGTGGGCCTTGCTGGGTGGGGCAGTGCTAGCGAGGCCGGCGGGCAGGGGAAGAGGGTGGGCACTGGGGGCAGAGAGAACTGCTTAGCGAAGGTAAGGTACGAGGAGGCAAACACATAAGGC', 'CTTCCTGCGGCGCAAGCTGCGCACGTGGGCCTTGCTGGGTGGGGCAGTGCTAGCGAGGCCGGCGGGCAGGGGAAGAGGGTGGGCACTGGGGGCAGAGAGAACTGCTTAGCGAAGGTAAGGTACGAGGAGGCAAACACATAAGGC', 'AAAAGCCTCGGTCGCAGCACCAGTCTCTCCATCTTCTTCAAAGGTGCCTTACCTTTCTTATTCCAAAAATGGCTGGGCCACAAGGCCCAAACCAAGAGAGATCAGCCCCAGCACAAGACCCCGAAGGCCACTCAGCATCTTCCT', 'GCCTGTAATCCCAGCACTTTGGGTGGCTGAGGCAGGCAGATCACGAGGTCAGGAGATCGAGACCATCCTGGCTAACACGGTGAAACCCCGTCTCTACTAAAAATGCAAAAAAAATTCGCTGGGCGTGGTGGCGGGCGCCTGTAG', 'AGATGAAACAGCCTATTGAAAGAAGATGTCATCTAGAACTTTCATAGCTAGAGAAGAGAAGTCAATGCCTAACCTCAGATCTTCAAACAACAGGCTGACTCTATTGTTAGTAACTAATGCAGCTAGTGACTTCAGTTGAAGCCA']
for seq in seqs:
    mut = [seq, seq[:71] + 'T' + seq[72:]]
    X = seqs_to_encoded_matrix(mut)
    pred = model.predict(X)
    print pred[0, :] - pred[1, :]

def load_fasta(file):
    seqs = []
    with open(file) as f:
        for line in f:
            if line[0] == '>': continue
            seqs += [line.strip().upper()]
    return seqs

test = load_fasta('../data/test.fa')
back = load_fasta('../data/background.fa')

test_pred = model.predict(seqs_to_encoded_matrix(test))
back_pred = model.predict(seqs_to_encoded_matrix(back))

print test_pred.mean()
print back_pred.mean()

def mutate(seq):
    rev = {'A': 'G',
          'T': 'A',
          'C': 'T',
          'G': 'C'}
    return seq[:71] + rev[seq[71]] + seq[72:]

test_mut = model.predict(seqs_to_encoded_matrix(map(mutate, test)))
back_mut = model.predict(seqs_to_encoded_matrix(map(mutate, back)))

print test_pred.mean(axis = 0)
print back_pred.mean(axis = 0)

print test_mut.mean(axis = 0)
print back_mut.mean(axis = 0)

print np.abs(test_mut - test_pred).mean(axis = 0)
print np.abs(back_mut - back_pred).mean(axis = 0)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.hist((test_mut - test_pred)[:, 0], bins = 100)
plt.show()

plt.hist((back_mut - back_pred)[:, 0], bins = 100)
plt.show()

print np.abs(test_mut - test_pred)[10000:].mean(axis = 0)

print np.abs(test_mut - test_pred)[:10000].mean(axis = 0)

def quantile_plot(data_pairs, quantiles = 100):
    data_pairs = sorted(data_pairs, key = lambda x: x[0])
    avg_activity, avg_score = [], []
    for i in range(0, len(data_pairs), len(data_pairs) / quantiles):
        index = range(i, min(i + (len(data_pairs) / quantiles), len(data_pairs)))
        activities = [data_pairs[j][0] for j in index]
        scores  = [data_pairs[j][1] for j in index]
        avg_activity.append(sum(activities) / float(len(activities)))
        avg_score.append(sum(scores) / float(len(scores)))
    i = 0
    plt.scatter(avg_activity, avg_score, c = 'r')

plt.scatter(range(test_mut.shape[0]), np.abs(test_mut - test_pred)[:, 0])
quantile_plot(enumerate(np.abs(test_mut - test_pred)[:, 0]))
plt.show()



