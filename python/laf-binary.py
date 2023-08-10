ROOT = '/Users/dirk/laf/laf-fabric-data/etcbc4b/bin'
FILE = 'Fn0(etcbc4,ft,lex)'

import pickle, gzip

with gzip.open('{}/{}'.format(ROOT, FILE), "rb") as f:
    data = pickle.load(f)

print('type: {}'.format(type(data)))

print('{} keys'.format(len(data)))

print('\n'.join('{:>6}: "{}"'.format(*x) for x in sorted(data.items())[0:20]))

print('\n'.join(x[1] for x in sorted(data.items())[0:20]))

print('\n'.join(x[1] for x in sorted(data.items())[200000:200020]))



