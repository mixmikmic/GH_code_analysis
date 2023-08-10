from typing import Tuple

def two_sum_sorted(numbers: list, target: int) -> Tuple[int, int]:
    """
    Uses two pointers on the array. If the two numbers being pointed to 
    sum lower than the target, then increase the index of the "lower" pointer. 
    If they sum higer than the target, then decrease the index of the "higher"
    pointer.
    """
    
    if not numbers: #handles empty lists
        return None
    
    i = 0
    j = len(numbers) - 1
    
    while i < j:
        x = numbers[i] + numbers[j]
        if x < target:
            i += 1
        elif x > target:
            j -= 1
        else:
            return i, j
    
    return None

assert two_sum_sorted([2,7,11,15], 9) == (0, 1)
assert two_sum_sorted([2,7,11,15], 26) == (2, 3)
assert two_sum_sorted([2,7,11,16], 26) == None
assert two_sum_sorted([], 26) == None

