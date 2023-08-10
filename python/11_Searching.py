import bisect
import math

def binary_search(seq, x):
    """
    Performs binary search for sequences of primitive types
    """
    lo, hi = 0, len(seq) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if x > seq[mid]:
            lo = mid + 1
        elif x == seq[mid]:
            return mid
        else:
            hi = mid - 1
    return None


def binary_search_pythonic(seq, x):
    """
    Performs binary search using the `bisect` module
    """
    i = bisect.bisect_left(seq, x)
    if 0 <= i < len(seq) and seq[i] == x:
        return i
    else:
        None

def binary_search_first(A, k):
    """
    Returns the first occurence of `k` in `A` if present, else None
    """
    lo, hi, result = 0, len(A) - 1, None
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if k > A[mid]:
            lo = mid + 1
        elif k == A[mid]:
            result = mid
            hi = mid - 1
        else:
            hi = mid - 1
    return result

# Tests
assert binary_search_first([-14, -10, 2, 108, 108, 243, 285, 285, 401], 108) == 3
assert binary_search_first([-14, -10, 2, 108, 108, 243, 285, 285, 401], 285) == 6
assert binary_search_first([1, 1, 1], 1) == 0

def binary_search_first_pythonic(A, k):
    """
    Returns the first occurence of `k` in `A` if present, else None
    """
    i = bisect.bisect_left(A, k)
    if 0 <= i < len(A):
        return i
    else:
        None
    
# Tests
assert binary_search_first_pythonic([-14, -10, 2, 108, 108, 243, 285, 285, 401], 108) == 3
assert binary_search_first_pythonic([-14, -10, 2, 108, 108, 243, 285, 285, 401], 285) == 6
assert binary_search_first_pythonic([1, 1, 1], 1) == 0

# Variant 1: Design an efficient algorithm that takes a sorted array and a key, and finds the first occirence of an
# element greater than the key,
def binary_search_last(A, k):
    """
    Returns the first occurence of `k` in `A` if present, else None
    """
    lo, hi, result = 0, len(A) - 1, None
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if k > A[mid]:
            lo = mid + 1
        elif k == A[mid]:
            result = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return result

def binary_search_last_pythonic(A, k):
    """
    Returns the first occurence of `k` in `A` if present, else None
    """
    i =  bisect.bisect_right(A, k)
    if A[i - 1] == k:
        return i - 1
    else:
        return None
        
        
# Tests
assert binary_search_last([-14, -10, 2, 108, 108, 243, 285, 285, 401], 108) == 4
assert binary_search_last([-14, -10, 2, 108, 108, 243, 285, 285, 401], 285) == 7
assert binary_search_last([1, 1, 1], 1) == 2

assert binary_search_last_pythonic([-14, -10, 2, 108, 108, 243, 285, 285, 401], 108) == 4
assert binary_search_last_pythonic([-14, -10, 2, 108, 108, 243, 285, 285, 401], 285) == 7
assert binary_search_last_pythonic([1, 1, 1], 1) == 2

# Variant 2: Let A be an unsorted array of `n` integers, with A[0] >= A[1] and A[n - 1] <= A[n - 1]. Call an index `i`
# a local minimum if A[i] is less than or equal to its neighbours. How would you efficiently find a local minimum
# if it exists?

# Variant 3: Write a program that takes a sorted array A of integers, and an integer k, and returns the interval enclosing
# k i.e the pair of integers L and U such that L is the first occurence of k in A and U is the last occurence of k in A.
# If k does not appear in A, return [-1, -1]

def interval(A, k):
    return [binary_search_first(A, k), binary_search_last(A, k)]

# Tests
A = [1, 2, 2, 4, 4, 4, 7, 11, 11, 13]
assert interval(A, 11) == [7, 8]

# Variant 4: Write a program to test if `p` is a prefix of a string in an array of sorted strings

def str_prefix(A, p):
    """
    Returns index of array if prefix, else None
    """
    lo, hi = 0, len(A) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if A[mid].startswith(p):
            return mid
        elif A[mid] < p:
            lo = mid + 1
        else:
            hi = mid - 1
    return None

# Tests
A = ['aac', 'bba', 'ssa']
assert str_prefix(A, 'a') == 0

def cyclic_smallest(A):
    """
    Given a sorted array of distinct integers
    Returns the index of smallest element of a cyclically sorted array
    """
    lo, hi = 0, len(A) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if A[mid] > A[hi]:
            lo = mid + 1
        elif A[mid] < A[hi]:
            hi = mid
    return lo


# Tests
assert cyclic_smallest([378, 478, 550, 631, 103, 203, 220, 234, 279, 368]) == 4

def int_sqroot(n):
    """
    Given a non-neg int `n`
    Returns the largest integer whose square is less than or equal to `n`
    """
    lo, hi, result = 0, n, None
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        mid_sqr = mid * mid
        
        if mid_sqr == n:
            return mid
        elif mid_sqr < n:
            lo = mid + 1
            result = mid
        else:
            hi = mid - 1
    return result
            

# Tests
assert int_sqroot(16) == 4
assert int_sqroot(21) == 4
assert int_sqroot(300) == 17

def real_sqroot(n):
    """
    Given a non-neg real number `n`
    Returns the largest number whose square is less than or equal to `n`
    """
    lo, hi = (n, 1.0) if n < 1.0 else (1.0, n)
    while not math.isclose(lo, hi):
        mid = 0.5 * (lo + hi)
        if mid*mid > n:
            hi = mid
        else:
            lo = mid
    return lo

# Tests
assert math.isclose(real_sqroot(0.25), 0.5)

import random

def find_kth_largest(A, k):
    """
    Returns the kth largest elements of A
    """
    def partition_around_pivot(left, right, pivot_id):
        """
        Partitions A around pivot_id and returns the new pivot id
        """
        pivot_value = A[pivot_id]
        new_pivot_id = left
        A[pivot_id], A[right] = A[right], A[pivot_id]
        for i in range(left, right):
            if A[i] > pivot_value:
                A[new_pivot_id], A[i] = A[i], A[new_pivot_id]
                new_pivot_id += 1
        A[new_pivot_id], A[right] = A[right], A[new_pivot_id]
        return new_pivot_id
    
    left, right = 0, len(A) - 1
    while left <= right:
        pivot_id = random.randint(left, right)
        new_pivot_id = partition_around_pivot(left, right, pivot_id)
        if new_pivot_id == k - 1:
            return A[new_pivot_id]
        elif new_pivot_id < k - 1:
            left = new_pivot_id + 1
        else:
            right = new_pivot_id - 1
    

# Tests
A = [3, 1, -1, 2]
assert find_kth_largest(A, 1) == 3
assert find_kth_largest(A, 2) == 2
assert find_kth_largest(A, 3) == 1
assert find_kth_largest(A, 4) == -1

import math

# Variant 1: Design an algorithm to find the median of an array
def mean_of_2(a, b):
    """
    Returns the mean of `a` and `b`
    """
    return 0.5 * (a + b)


def median(A):
    """
    Returns the median of the given array
    """
    if len(A) % 2 == 0:
        return mean_of_2(A[find_kth_largest(A, len(A)/2)], 
                         A[find_kth_largest(A, len(A)/2 + 1)])
    else:
        return A[find_kth_largest(A, math.ceil(len(A)/2))]

# Tests
B = [3, 1, -1, 2]
assert median(B) == 1.5
B.append(5)
assert median(B) == 2

from collections import namedtuple
from functools import reduce


DuplicateAndMissing = namedtuple('DuplicateAndMissing', ('duplicate', 'missing'))

def find_duplicate_missing(A):
    """
    Returns a tuple of a duplicate and missing element from the given sequence
    """
    miss_dup_xor = functools.reduce(lambda acc, x: acc ^ x[0] ^ x[1],
                                   enumerate(A),
                                   0)
    
    differ_bit = miss_dup_xor & ~(miss_dup_xor - 1)
    miss_or_dup = 0
    
    for i, a in enumerate(A):
        if i & differ_bit:
            miss_or_dup ^= i
        if a & differ_bit:
            miss_or_dup ^= a
    
    # miss_or_dup = dup
    if miss_or_dup in A:
        return DuplicateAndMissing(miss_or_dup, miss_or_dup ^ miss_dup_xor)
    # miss_or_dup = miss
    else:
        return DuplicateAndMissing(miss_or_dup ^ miss_dup_xor, miss_or_dup)

# Tests
A = [5, 3, 0, 3, 2, 1]
assert find_duplicate_missing(A).duplicate == 3
assert find_duplicate_missing(A).missing == 4

