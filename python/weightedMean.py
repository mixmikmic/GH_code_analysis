def weighted_mean(N, Xs, Ws):
    weighted_sum = 0
    weight_sum = 0
    for i in range(N):
        weighted_sum += Xs[i] * Ws[i]
        weight_sum += Ws[i]
    return round(weighted_sum / weight_sum, 1)

import unittest

class TestWeightedMean(unittest.TestCase):
    def setUp(self):
        self.N = 5
        self.Xs = [10, 40, 30, 50, 20]
        self.Ws = [1, 2, 3, 4, 5]
    
    def test_weighted_mean(self):
        res = weighted_mean(self.N, self.Xs, self.Ws)
        self.assertEqual(res, 32)

if __name__ == '__main__':
    unittest.main(argv=['Ignore first argument'], exit=False)



