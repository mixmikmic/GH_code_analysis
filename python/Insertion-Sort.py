from resources.utils import run_tests

def insertion_sort(input_list):
    def swap(i, j):
        input_list[i], input_list[j] = input_list[j], input_list[i]
    
    cursor = 1
    
    while cursor < len(input_list):
        if input_list[cursor] < input_list[cursor - 1]:
            swap(cursor, cursor - 1)
            if cursor > 1:
                cursor -= 1
            else:
                cursor += 1
        else:
            cursor += 1
                
    return input_list

tests = [
    ({'input_list': [3,2,1]}, [1,2,3]),
    ({'input_list': []}, []),
    ({'input_list': [1,2]}, [1,2]),
    ({'input_list': [5,1,4,2,3]}, [1,2,3,4,5]),
    ({'input_list': []}, []),
]

run_tests(tests, insertion_sort)

