import random
random.seed(0)
from resources.utils import run_tests

def split(input_list):
    """
    Splits a list into two pieces
    :param input_list: list
    :return: left and right lists (list, list)
    """
    input_list_len = len(input_list)
    midpoint = input_list_len // 2
    return input_list[:midpoint], input_list[midpoint:]

tests_split = [
    ({'input_list': [1, 2, 3]}, ([1], [2, 3])),
    ({'input_list': [1, 2, 3, 4]}, ([1, 2], [3, 4])),
    ({'input_list': [1, 2, 3, 4, 5]}, ([1, 2], [3, 4, 5])),
    ({'input_list': [1]}, ([], [1])),
    ({'input_list': []}, ([], []))
]

run_tests(tests_split, split)

def merge_sorted_lists(list_left, list_right):
    """
    Merge two sorted lists
    This is a linear operation
    O(len(list_right) + len(list_right))
    :param left_list: list
    :param right_list: list
    :return merged list
    """
    # Special case: one or both of lists are empty
    if len(list_left) == 0:
        return list_right
    elif len(list_right) == 0:
        return list_left
    
    # General case
    index_left = index_right = 0
    list_merged = []  # list to build and return
    list_len_target = len(list_left) + len(list_right)
    while len(list_merged) < list_len_target:
        if list_left[index_left] <= list_right[index_right]:
            # Value on the left list is smaller (or equal so it should be selected)
            list_merged.append(list_left[index_left])
            index_left += 1
        else:
            # Right value bigger
            list_merged.append(list_right[index_right])
            index_right += 1
            
        # If we are at the end of one of the lists we can take a shortcut
        if index_right == len(list_right):
            # Reached the end of right
            # Append the remainder of left and break
            list_merged += list_left[index_left:]
            break
        elif index_left == len(list_left):
            # Reached the end of left
            # Append the remainder of right and break
            list_merged += list_right[index_right:]
            break
        
    return list_merged

tests_merged_sorted_lists = [
    ({'list_left': [1, 5], 'list_right': [3, 4]}, [1, 3, 4, 5]),
    ({'list_left': [5], 'list_right': [1]}, [1, 5]),
    ({'list_left': [], 'list_right': []}, []),
    ({'list_left': [1, 2, 3, 5], 'list_right': [4]}, [1, 2, 3, 4, 5]),
    ({'list_left': [1, 2, 3], 'list_right': []}, [1, 2, 3]),
    ({'list_left': [1], 'list_right': [1, 2, 3]}, [1, 1, 2, 3]),
    ({'list_left': [1, 1], 'list_right': [1, 1]}, [1, 1, 1, 1]),
    ({'list_left': [1, 1], 'list_right': [1, 2]}, [1, 1, 1, 2]),
    ({'list_left': [3, 3], 'list_right': [1, 4]}, [1, 3, 3, 4]),
]

run_tests(tests_merged_sorted_lists, merge_sorted_lists)

def merge_sort(input_list):
    if len(input_list) <= 1:
        return input_list
    else:
        left, right = split(input_list)
        # The following line is the most important piece in this whole thing
        return merge_sorted_lists(merge_sort(left), merge_sort(right))

random_list = [random.randint(1, 1000) for _ in range(100)]
tests_merge_sort = [
    ({'input_list': [1, 2]}, [1, 2]),
    ({'input_list': [2, 1]}, [1, 2]),
    ({'input_list': []}, []),
    ({'input_list': [1]}, [1]),
    ({'input_list': [5, 1, 1]}, [1, 1, 5]),
    ({'input_list': [9, 1, 10, 2]}, [1, 2, 9, 10]),
    ({'input_list': range(10)[::-1]}, list(range(10))),
    ({'input_list': random_list}, sorted(random_list))
]

run_tests(tests_merge_sort, merge_sort)

