# Bad Fib - recursive algorithm that takes longer the higher the humber is

# Create a socket server!

# listen for a request, and handle it by calling fib and writing it into the response

# Let's test performance!
import time

# Create a connection

while True:
        start = time.time()
        # make a request and get response
        end = time.time()
        print end

# Another test for requests/second
import time

# Create a connection

#make a global variable (counter) and a thread to monitor it

while True:
        start = time.time()
        # make a tiny request and get response
        end = time.time()
        print end

# Threads suck, pools are 'meh'  let's try something else


from collections import dequeue
tasks = dequeue()

# put some tasks in there

# Round robin scheduler
while tasks:
    t = tasks.popleft()
    try:
            x = next(t)
            print x
            tasks.append(x)
    except StopIteration:
        print 'done!'
    

# put yield statements into code where blocking things happen

# yield returns "what" and "why"

# check the reason, and shove waiting tasks into an waiting area (penalty box)

# While there are any tasks in queue, or waiting areas

import select

# This teslls us who can send or who can recieve (from the OS)

# Append the results from select back into the tasks queue





