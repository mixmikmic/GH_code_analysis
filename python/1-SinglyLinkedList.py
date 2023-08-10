class ListNode:
    def __init__(self, data):
        self.data = data
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None
    
    # Append data to the end of the linked list; naive approach
    def append(self, data):
        node = ListNode(data)
        if self.head is None:
            self.head = node
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            current.next = node

linked_list = SinglyLinkedList()
linked_list.append("egg")
linked_list.append("ham")
linked_list.append("spam")
current = linked_list.head
while current:
    print(current.data) 
    current = current.next
del linked_list

class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    
    def append(self, data):
        node = ListNode(data)
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node

class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0  # Add a size variable
        
    def append(self, data):
        node = ListNode(data)
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self.size += 1  # Add one whenever an element is inserted

class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0  
        
    def append(self, data):
        node = ListNode(data)
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self.size += 1
    
    # Delete the first appearance of data from the list 
    def delete(self, data):
        current = self.head
        prev = None
        while current is not None:
            if current.data == data:
                if current == self.head:
                    self.head = self.head.next
                    if current == self.tail:
                        self.tail = tail.next
                else:
                    prev.next = current.next
                    if current == self.tail:
                        self.tail = prev
                self.size -= 1
                return 
            prev = current
            current = current.next
    
    # Get the iterator for the list
    def __iter__(self):
        current = self.head
        while current is not None:
            data = current.data
            current = current.next
            yield data

linked_list = SinglyLinkedList()
linked_list.append("egg")
linked_list.append("ham")
linked_list.append("spam")
for item in iter(linked_list):
    print(item)

linked_list.delete("egg")
linked_list.append("egg")
linked_list.append("bacon")
linked_list.delete("spam")
linked_list.delete("bacon")
for item in iter(linked_list):
    print(item)
print(linked_list.size)

# Final version
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0  
        
    def append(self, data):
        node = ListNode(data)
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self.size += 1
        
    def delete(self, data):
        current = self.head
        prev = None
        while current is not None:
            if current.data == data:
                if current == self.head:
                    self.head = self.head.next
                    if current == self.tail:
                        self.tail = tail.next
                else:
                    prev.next = current.next
                    if current == self.tail:
                        self.tail = prev
                self.size -= 1
                return 
            prev = current
            current = current.next
    
    # Search through the linked list to see if there exists a node with the given data 
    def search(self, data):
        current = self.head
        while current is not None:
            if current.data == data:
                return True
            current = current.next
        return False
    
    # Clear all nodes
    def clear(self):
        self.head = None
        self.tail = None
        
    def __iter__(self):
        current = self.head
        while current is not None:
            data = current.data
            current = current.next
            yield data

