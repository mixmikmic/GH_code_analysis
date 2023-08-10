from pynqhls.memmap import memmapOverlay
overlay = memmapOverlay('memmap.bit')

import numpy as np

A = np.random.randint(-10, 10, size=(10,10))
B = np.random.randint(-10, 10, size=(10,10))
C = np.matmul(A, B)

CHLS = overlay.run(A, B)

np.array_equal(CHLS, C)



