class Node(object):
    def __init__(self, data=None, next_node=None):
        """
        Initializes Node in Linked List.
        """
        self.data = data
        self.next = next_node


    def __str__(self):
        """
        Converts current linked list to a string.
        """
        node = self
        buffer = str(node.data)

        node = node.next
        while node != None:
            buffer += ' -> ' + str(node.data)
            node = node.next

        return buffer

class Stack(object):
    
    def __init__(self, top=None):
        self.top = top
        
    
    def push(self, data):
        
        if self.top == None: #Check if our stack is empty
            self.top = Node(data) #Make a new node 
            return
            
        temp = Node(data) #Create a node
        temp.next = self.top #Make our node point to the top node
        self.top = temp #Make our node the new top node
        
    def pop(self):
        
        if self.top == None: #Check if our stack is empty
            return None #return None if so
        
        temp = self.top #Save the top node
        self.top = temp.next #Make the new top node the next node
        return temp.data #Return our saved value
    
    def peek(self):
        
        if self.top == None: #Check if our stack is empty
            return None #return None if so
        
        return self.top.data
        

s = Stack()
s.push(1)
s.push(2)
s.push(3)
s.push(4)

print(s.top)

s.pop()

s.peek()

print(s.top)

class Queue(object):
    
    def __init__(self, front=None, back=None):
        self.front = front
        self.back = back
        
    
    def enqueue(self, data):
        
        if self.front == None: #Check if our stack is empty
            self.front = Node(data) #Make a new node 
            self.back = self.front #Make our pointers right
            return
            
        temp = Node(data) #Create a node
        self.back.next = temp #Place our node at the very back
        self.back = temp #Make our node the new back node
        
    def dequeue(self):
        
        if self.front == None: #Check if our stack is empty
            return None #return None if so
        
        temp = self.front #Save the top node
        self.front = temp.next #Make the new top node the next node
        return temp.data #Return our saved value
    
    

q = Queue()
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
q.enqueue(4)

print(q.front)

q.dequeue()

print(q.front)

