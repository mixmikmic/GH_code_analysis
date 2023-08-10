import numpy as np

# A 6x5 robot world
world = np.array([ [0, 0, 0, 1, 0],
                   [0, 0, 0, 1, 0],
                   [0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1],
                   [1, 0, 0, 1, 0],
                   [1, 0, 0, 0, 0] ])

# Print out some information about the world
print(world)
print('\nThe shape of this array is: ' + str(world.shape))

# This function uses nested for loops and knowledge
# about the shape of the array to print out each item with it's index
def iterate2D(world):
    # y-dimension (rows)
    for i in range(0, world.shape[0]):
        # x-dimension (columns)
        for j in range(0, world.shape[1]):
            print('Index ['+str(i)+']['+str(j)+'] = ' +str(world[i][j]))

# Call the iterate function
print('\n')
iterate2D(world)

# This function is similar to our iterate2D function,
# But looks for the first tree in the array and prints its location [x][y]
def first_tree(world):
    # iterates through all indices starting at the top-left [0][0]
    for i in range(0, world.shape[0]):
        for j in range(0, world.shape[1]):
            # check if a tree is found
            if(world[i][j] == 1):
                # if so, print the index and leave the loop with a return statement
                print('First tree found at location: ['+str(i)+']['+str(j)+']')
                return 

            
# Call the first_tree function
print('\n')
first_tree(world)



