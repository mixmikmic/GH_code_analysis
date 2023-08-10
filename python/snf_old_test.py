from snf_old import Smith

import numpy as np

from datetime import datetime

matrix = np.random.random_integers(0, 1, (120,120))
matrix = matrix.tolist()



times = []
for i in range(20):
    matrix = np.random.random_integers(0, 1, (120,120))
    matrix = matrix.tolist()
    start = datetime.now()
    Smith(matrix)
    end = datetime.now()
    duration = (end - start).total_seconds()
    print(duration)
    times.append(duration)

np.mean(times)

