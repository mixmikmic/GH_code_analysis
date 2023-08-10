def print_vec(x, y, z): 
    print ('<{}, {}, {}>'.format(x, y, z))
    
print_vec(2, 4, 6)

my_tuple = (2, 4, 6)

#need to unpack each value of the tuple to print... :(
print_vec(my_tuple[0], my_tuple[1], my_tuple[2])

print_vec(*my_tuple)

my_list = [3, 9, 6]

print_vec(*my_list)

#can we use the function unpacker by itself, without a function?
*my_list

#what happens if we unpack too many values for the function input?

long_list = [5, 23, 7, 9, 3, 8]

print_vec(*long_list)

#but maybe we can truncate the excess?
print_vec(*long_list[:3])

d = {'x': 1, 
     'y': 1, 
     'z': 4}

print_vec(**d)

print_vec(*d)



