class Node(object):
    
    def __init__(self,value):
        
        self.value = value
        self.nextnode = None

def reverse(head):
    
    # Take care of head pointing to None edge case
    if head.nextnode == None:
        return head
    
    # set up variables to current and previous nodes
    current = head
    previous = None
    next_node = None
    #print current.nextnode.value
    
    # Travers list till tail is found, reversing along the way
    while current:
        
        # First, update next_node
        # This is KEY!  if done out of sequence, pointers will be circular
        next_node = current.nextnode
        
        # Flip current node to point to previous
        current.nextnode = previous
        
        # Update current and previous
        previous = current 
        current = next_node
        
    return previous      

# Create a list of 4 nodes
a = Node(1)
b = Node(2)
c = Node(3)
d = Node(4)

# Set up order a,b,c,d with values 1,2,3,4
a.nextnode = b
b.nextnode = c
c.nextnode = d

print a.value
print a.nextnode.value
print b.nextnode.value
print c.nextnode.value

if d.nextnode == None:
    print "Yes, the tail points to None as expected"

reverse(a)

print d.value
print d.nextnode.value
print c.nextnode.value
print b.nextnode.value

if a.nextnode == None:
    print "Yes, the head is now the tail!!"



