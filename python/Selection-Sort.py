from resources.utils import run_tests

def selection_sort(input_list):
    def replace_min(current, index):
        min_ = current
        min_index = index - 1
        for i, val in enumerate(input_list[index:]):
            if val < min_:
                min_ = val
                min_index = index + i
            
        if min_ != current:
            # swap
            input_list[index -1], input_list[min_index] = input_list[min_index], input_list[index -1]
    
    for index in range(len(input_list)):
        replace_min(input_list[index], index + 1)
            
    return input_list

tests = [
    ({'input_list': [5,4,3,2,1]}, [1,2,3,4,5]),
    ({'input_list': [1,10,2,9,3,5]}, [1,2,3,5,9,10]),
    ({'input_list': [1]}, [1]),
    ({'input_list': [3,1]}, [1,3]),
    ({'input_list': [1000,-1,0]}, [-1,0,1000]),
    ({'input_list': [1,2,100]}, [1,2,100]),
]

run_tests(tests, selection_sort)

