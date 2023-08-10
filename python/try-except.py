def special_func(test_case_dict):
    final_dict = {}
    exception_dict = {}

    def try_except_avoider(test_case_dict):

        try:
            for k,v in test_case_dict.items():
                final_dict[k]=eval(v) #If no exception evaluate the function and add it to final_dict

        except Exception as e:
            exception_dict[k]=e #extract exception
            test_case_dict.pop(k)
            try_except_avoider(test_case_dict) #recursive function to handle remaining functions

        finally:  #cleanup
            final_dict.update(exception_dict)
            return final_dict #combine exception dict and  final dict
        
    return try_except_avoider(test_case_dict)

def add(a,b):
    return (a+b)
def sub(a,b):
    return (a-b)
def mul(a,b):
    return (a*b)

case = {"AddFunc":"add(9,8)","SubFunc":"sub(8,5)","MulFunc":"mul(,6)"}
solution = special_func(case)
solution

locals().update(solution)

AddFunc

MulFunc

SubFunc

