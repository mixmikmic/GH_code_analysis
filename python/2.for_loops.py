def small_list_v1(one, two, three):
    list = [one, two, three]
    print list

def small_list_v2(one, two, three):
    list = []
    list.append(one)
    list.append(two)
    list.append(three)
    print list

def small_list_v3(num):
    list = range(0, num)
    print list
    

def unknown_function(unknown_parameter):
    unknown_variable = range(0, unknown_parameter)
    return unknown_variable

def unknown_function2():
    unknown_variable = unknown_function(5)
    print unknown_variable

def unknown_function3(given_list):
    for val in given_list:
        print val

def unknown_function4():
    unknown_variable = range(0, 6)
    unknown_variable2 = []
    for i in unknown_variable:
        unknown_variable2.append(i)
    return unknown_variable2

def unknown_function5(words_list):
    f = open("words_list.txt", "w")
    for word in words_list:
        f.write(word)
    f.close()

def unknown_function6(filename):
    f = open(filename, "r")
    for line in f:
        print line
    f.close()
        

