largest = 0

for i in range(100,1000):
    for j in range(100,1000):
        int_num = i*j
        str_num = str(int_num)
        if str_num == str_num[::-1] and int_num > largest:
            largest = int_num
            
print(largest)

