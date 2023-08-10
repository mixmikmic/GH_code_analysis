class CentralTendency:
    def __init__(self, numbers, size=None):
        self.numbers = numbers
        self.size = len(numbers) if size is None else size
    
    @classmethod
    def quicksort(cls, array):
        if len(array) <= 1:
            return array
        pivot = array[-1]
        left, right = [], []
        for n in array[:-1]:
            if n < pivot:
                left.append(n)
            else:
                right.append(n)
        return cls.quicksort(left) + [pivot] + cls.quicksort(right)

    def mean(self):
        sum_ = 0
        for n in self.numbers:
            sum_ += n
        return sum_ / self.size

    def median(self):
        if self.size == 0:
            return None
        sorted_nums = CentralTendency.quicksort(self.numbers)
        middle_idx = (self.size - 1) // 2
        if self.size % 2 == 1:
            return sorted_nums[middle_idx]
        return (sorted_nums[middle_idx] + sorted_nums[middle_idx + 1]) / 2

    def mode(self):
        counter = {}
        for n in self.numbers:
            try:
                counter[n] += 1
            except KeyError:
                counter[n] = 1
        return max(sorted(counter.items()), key=lambda tuple_: tuple_[1])[0]

import unittest
import numpy as np
from scipy import stats

class TestCentralTendency(unittest.TestCase):
    def setUp(self):
        self.numbers = [2, 5, 7, 1, 3, 4, 7]
        self.ct = CentralTendency(self.numbers)
    
    def test_mean(self):
        self.assertEqual(self.ct.mean(), np.mean(self.numbers))

    def test_median(self):
        self.assertEqual(self.ct.median(), np.median(self.numbers))

    def test_mode(self):
        self.assertEqual(self.ct.mode(), int(stats.mode(self.numbers)[0]))

if __name__ == '__main__':
    unittest.main(argv=['Ignore first argument'], exit=False)

