print("1 - demonstrating for loop")
for i in [0,1,2,3,4,5]:
    print(i)

print("2 - demonstrating for loop with range()")
for i in range(6):
    print(i)

for left_num in range(5):
    for right_num in range(5):
        product = left_num * right_num
        print(left_num, "x", right_num, "=", product)

