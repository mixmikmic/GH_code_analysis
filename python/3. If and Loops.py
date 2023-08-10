x = 4

if x < 2: 
    x += 1
elif x == 3:
    x = x
else:
    x-=1
    
print(x)

# Add up the first numbers less than or equal to n

n = 5
ix = 1
total_sum = 0
while(ix <= n):
    total_sum += ix
    # This is super important, without it we would have an infinite loop
    # remove it and see what happens (if you do, click Kernel in the header above, and press Restart)
    ix += 1
    print('current total = ' + str(total_sum))
    
print(total_sum)

numbers = [1, 2, 3, 4, 5]

for i in numbers:
    print(i)

for i in range(1, 6):
    print(i)

for i in range(1, 6):
    print(i)
    if(i == 3):
        break

