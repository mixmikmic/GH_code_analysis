class ListNode:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    # Add data to the end of the list; update head, tail, and size
    def append(self, data):
        node = ListNode(data)
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
        self.size += 1
        
    def __iter__(self):
        current = self.head
        while current is not None:
            data = current.data
            current = current.next
            yield data

dll = DoublyLinkedList()
for color in ("yellow", "blue", "purple", "green"):
    dll.append(color)
for data in dll:
    print(data)
del dll

class DoublyLinkedList:
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
            node.prev = self.tail
            self.tail = node
        self.size += 1
    
    # Delete the first occurrance of data in the list 
    def delete(self, data):
        current = self.head
        while current is not None:
            if current.data == data:
                if current == self.head:  # Deletion at the beginning of the list
                    self.head = current.next  # Update head
                    if current == self.tail:  # Only a single element
                        self.tail = None
                    else:
                        self.head.prev = None
                elif current == self.tail:  # Deletion at the end of the list
                    current.prev.next = None
                    self.tail = current.prev  # Update tail
                else:  # Delection in the middle of the list
                    current.prev.next = current.next
                    current.next.prev = current.prev
                self.size -= 1
                return
            current = current.next
        
    def __iter__(self):
        current = self.head
        while current is not None:
            data = current.data
            current = current.next
            yield data

dll = DoublyLinkedList()
for color in ("yellow", "blue", "purple", "green", "black", "orange"):
    dll.append(color)
dll.delete("yellow")
dll.delete("green")
dll.delete("orange")
for data in dll:
    print(data, end=" ")
print("\nHead: {}".format(dll.head.data))
print("Tail: {}".format(dll.tail.data))
del dll

class DoublyLinkedList:
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
            node.prev = self.tail
            self.tail = node
        self.size += 1
    
    def delete(self, data):
        current = self.head
        while current is not None:
            if current.data == data:
                if current == self.head:  # Deletion at the beginning of the list
                    self.head = current.next  # Update head
                    if current == self.tail:  # Only a single element
                        self.tail = None
                    else:
                        self.head.prev = None
                elif current == self.tail:  # Deletion at the end of the list
                    current.prev.next = None
                    self.tail = current.prev  # Update tail
                else:  # Delection in the middle of the list
                    current.prev.next = current.next
                    current.next.prev = current.prev
                self.size -= 1
                return
            current = current.next
    
    # Search to see if the list contains data
    def search(self, data):
        for value in iter(self):
            if data == value:
                return True
        return False
    
    # Clear the whole list
    def clear(self):
        self.head = None
        self.tail = None
    
    def __iter__(self):
        current = self.head
        while current is not None:
            data = current.data
            current = current.next
            yield data

dll = DoublyLinkedList()
for color in ("yellow", "blue", "purple", "green", "black", "orange"):
    dll.append(color)
print(dll.search("purple"))
print(dll.search("PurPLE"))
del dll

