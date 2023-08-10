# takes a step and moves on
def move((x, y), direction):
    if direction == "up":
        y += 1
    if direction == "down":
        y -= 1
    if direction == "right":
        x += 1
    if direction == "left":
        x -= 1
    return (x, y)

# decides on direction
def change(index):
    if index > 2:
        return 0
    else: 
        return index+1

# input n number of steps and output coordinates
def spiral(n, directions=["up", "right", "down", "left"]):
    ind = 0
    s = steps_taken = 0
    step_level = 1
    x = y = 0

    if n == 0:
        return ()
    elif n == 1:
        return (0, 0)
    elif n == 2:
        return (0, 1)
    else:
        # print "{}".format(n)
        while steps_taken < n:
            
            # keep moving in direction until step level reached
            while s < step_level:
                (x, y) = move((x, y), directions[ind])
                steps_taken += 1
                s += 1
                
                # print "{}: {} \t [{} steps of {} {}]".format(steps_taken, (x, y), s, step_level, directions[ind])

                if steps_taken == n:
                    return (x, y)

            # reset step counter, if right/left add incr step level
            s = 0
            if ((ind == 1) or (ind == 3)): step_level += 1

            # reaching step level means changing direction
            ind = change(ind)

    return (x, y)

#test the output for a few use cases

for x in range(3, 7):
    print "{} steps: {}".format(x, spiral(x))



