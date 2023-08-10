import datetime

import numpy as np

input_matrices = [
    np.random.uniform(low=-100., high=100., size=(300,300)) for _ in range(4)
]

before = datetime.datetime.utcnow()

inverses = []
for matrix in input_matrices:
    inverses.append(np.linalg.inv(matrix))

result = sum(M[50][50] for M in inverses)

after = datetime.datetime.utcnow()
elapsed = (after - before).total_seconds()

print('Inverted 4 300x300 matrices.')
print('Sum of M^-1[50][50] is {}'.format(result))
print('Time elapsed: {0:.3f}s'.format(elapsed))

np.show_config()

