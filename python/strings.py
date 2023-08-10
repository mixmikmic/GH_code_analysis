s = "Hello"

s[0]

s[0] = "J"

from collections import Counter

def number_needed(a, b):
    set_a = set(a)
    set_b = set(b)
    not_in_b = set_a.difference(set_b)
    not_in_a = set_b.difference(set_a)
    in_both = set_a.intersection(set_b)
    a_counts, b_counts = Counter(a), Counter(b)
    del_count = 0
    for b_ltr in not_in_a:
        del_count += b_counts[b_ltr]
    for a_ltr in not_in_b:
        del_count += a_counts[a_ltr]
    for ltr in in_both:
        del_count += abs(a_counts[ltr] - b_counts[ltr])
    return del_count

import unittest

class TestNumbersNeeded(unittest.TestCase):
    def test_four_deletions(self):
        self.assertEqual(number_needed('cde', 'abc'), 4)

if __name__ == '__main__':
    unittest.main(argv=['Ignore first argument'], exit=False)



