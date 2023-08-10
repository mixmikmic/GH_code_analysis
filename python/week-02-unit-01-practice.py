







string = "The quick brown fox jumps over a lazy dog."

str_list = string.split()

str_list

for i in range(1, len(str_list), 2):
    print(str_list[i])

' '.join(str_list)

