def mean(array):
    sum_ = 0
    for val in array:
        sum_ += val
    return sum_ / len(array)

def stdev(array):
    mu = mean(array)
    squared_residuals = 0
    for val in array:
        squared_residuals += (val - mu)**2
    return round((squared_residuals / len(array))**0.5, 1)

import unittest

class TestStandardDeviation(unittest.TestCase):
    def setUp(self):
        self.numbers = [10, 40, 30, 50, 20]
    
    def test_standard_deviation(self):
        self.assertEqual(stdev(self.numbers), 14.1)

if __name__ == '__main__':
    unittest.main(argv=['Ignore first argument'], exit=False)



