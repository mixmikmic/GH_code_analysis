num_tests_1 = 0

for k in range(1, 31):

    if k % 15 == 0:
        num_tests_1 += 1
        print(k, 'fizzbuzz')

    elif k % 3 == 0:
        num_tests_1 += 1
        print(k, 'fizz')

    elif k % 5 == 0:
        num_tests_1 += 1
        print(k, 'buzz')

print('num_tests_1 =', num_tests_1)

num_tests_2 = 0

for k in range(1, 31):

    if k % 3 == 0:
        num_tests_2 += 1
        print(k, 'fizz')

    elif k % 15 == 0:
        num_tests_2 += 1
        print(k, 'fizzbuzz')

    elif k % 5 == 0:
        num_tests_2 += 1
        print(k, 'buzz')

print('num_tests_2 =', num_tests_2)

num_tests_3 = 0

for k in range(1, 31):

    if k % 3 == 0:
        num_tests_3 += 1

        if k % 5 == 0:
            num_tests_3 += 1
            print(k, 'fizzbuzz')

        else:
            print(k, 'fizz')
   
    elif k % 5 == 0:
        num_tests_3 += 1
        print(k, 'buzz')

print('num_tests_3 =', num_tests_3)

