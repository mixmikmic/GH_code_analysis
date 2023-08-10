import numpy as np
import random

from skbio.alignment import global_pairwise_align_nucleotide
from skbio import DNA

sequences = []
num_sequences = 50
mean_biological_sequence_length = 400
std_biological_sequence_length = 40
mean_nonbiological_sequence_length = 25
std_nonbiological_sequence_length = 2
# imagine that we have four slightly different reverse primers
reverse_primers = [DNA("ACCGTCGACCGTTAGGATA"),
                   DNA("ACCGTGGACCGTGAGGATT"),
                   DNA("ACCGTCGACCGTTAGGATT"),
                   DNA("ACCGTGGACCGTGAGGATG")]

for i in range(num_sequences):
    # determine the length for the current biological sequence. if it's less than 1, make the length 0
    biological_sequence_length = int(np.random.normal(mean_biological_sequence_length,
                                                      std_biological_sequence_length))
    if biological_sequence_length < 1:
        biological_sequence_length = 0
    # generate a random sequence of that length
    biological_sequence = ''.join(np.random.choice(list('ACGT'),biological_sequence_length))
    
    
    # determine the length for the current non-biological sequence. if it's less than 1, make the length 0
    non_biological_sequence_length = int(np.random.normal(mean_nonbiological_sequence_length,
                                                          std_nonbiological_sequence_length))
    if non_biological_sequence_length < 1:
        non_biological_sequence_length = 0
    # generate a random sequence of that length
    non_biological_sequence = ''.join(np.random.choice(list('ACGT'), non_biological_sequence_length))

    # choose one of the four reverse primers at random
    reverse_primer = random.choice(reverse_primers)
    
    # construct the observed sequence as the biological sequence, followed by the primer, followed by the 
    # non-biological sequence
    observed_sequence = ''.join(map(str, [biological_sequence, reverse_primer, non_biological_sequence]))
    seq_id = "seq%d" % i
    
    # append the result to the sequences list
    sequences.append(DNA(observed_sequence, metadata={'id': seq_id}))

print(repr(sequences[0]))

aln = global_pairwise_align_nucleotide(reverse_primers[0], sequences[0])[0]

gap_vector = aln[0].gaps()
primer_start_index = (~gap_vector).nonzero()[0][0]
print(primer_start_index)

print(sequences[0][:primer_start_index])

trimmed_sequences = []

for sequence in sequences:
    aln = global_pairwise_align_nucleotide(reverse_primers[0], sequence)[0]
    gap_vector = aln[0].gaps()
    primer_start_index = (~gap_vector).nonzero()[0][0]
    trimmed_sequences.append(DNA(sequence[:primer_start_index], metadata={'id': sequence.metadata['id']}))

import skbio

print("".join(skbio.io.write((s for s in trimmed_sequences), into=[], format='fasta')))

