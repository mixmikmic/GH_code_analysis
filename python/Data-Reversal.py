import numpy as np
import os

FILE = "data/cullpdb+profile_6133_filtered.npy.gz"
data = np.load(FILE)
# reshape to 700 residues by 57-length feature vectors;
# -1 denotes "whatever dimension fits here" for variable number of proteins
data = data.reshape(-1, 700, 57)

start = 0
end = len(data)

new_data = np.zeros_like(data)
for i in range(start, end):
    protein = data[i]
    # two cases: with or without 'NoSeq' padding
    # without padding - last residue is not 'NoSeq':
    if protein[-1][21] == 0:
        # reverse by iterating backwards
        new_data[i] = protein[::-1]
    # with padding - only reverse the valid residues, leave padding at end
    else:
        new_protein = [protein[i] for i in range(len(protein)-1, -1, -1) if protein[i][21] == 0]
        padding = [protein[i] for i in range(len(protein)) if protein[i][21] != 0]
        new_data[i] = np.vstack((new_protein, padding))

print(data[1][0])
print(data[1][-553])
print(new_data[1][-553])
print(new_data[1][0])

SAVE = "data/cullpdb+profile_6133_filtered_reversed.npy.gz"
save_data = np.vstack((new_data, data)).reshape(-1, 700*57)
with open(SAVE, 'wb') as f:
    np.save(f, save_data)



