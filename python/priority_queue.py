get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from priority_queue import BinaryHeap
import random

pq = BinaryHeap()

sample = random.sample(range(1, 20), 8)

for s in sample:
    pq.insert(s)
    print(pq.max(), pq.heap)

print('\nstart emptying the heap')
while not pq.is_empty():
    m = pq.delete_max()
    print(m, pq.heap)

