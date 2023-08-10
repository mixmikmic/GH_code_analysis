f = open("dq_unisex_names.csv", "r")
data = f.read()
print(data)

f = open('dq_unisex_names.csv', 'r')
data = f.read()
data_list = data.split("\n")
print(data_list[:5])

f = open('dq_unisex_names.csv', 'r')
data = f.read()
data_list = data.split('\n')

string_data = []
for data_elm in data_list:
    comma_list = data_elm.split(",")
    string_data.append(comma_list)
    
print(string_data[:5])

numerical_data = []
for str_elm in string_data:
    if len(str_elm) != 2:
        continue
    name = str_elm[0]
    num = float(str_elm[1])
    lst = [name, num]
    numerical_data.append(lst)
    
print(numerical_data[:5])

# The last value is ~100 people
numerical_data[len(numerical_data)-1]

thousand_or_greater = []
for num in numerical_data:
    if num[1] >= 1000:
        thousand_or_greater.append(num[0])
        
print(thousand_or_greater[:10])





