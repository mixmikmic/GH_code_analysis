class ListNode:
    """
    Represents a node of a singly linked list
    """
    def __init__(self, data=None):
        self.data = data
        self.next = None
    
# Linked list operations
def search_list(L, key):
    """
    Returns the list node that contains key. If not present, returns None
    """
    while L and L.data != key:
        L = L.next
    return L

def insert_after(node, new_node):
    """
    Inserts `new_node` after `node` in this list
    """
    assert node and new_node
    new_node.next = node.next
    node.next = new_node
    
def delete_after(node):
    """
    Deletes the node after `node`
    """
    assert node and node.next
    node.next = node.next.next
    
def seq_to_list(seq):
    """
    Given an iterable
    Returns it as a LinkedList
    """
    node = ListNode()
    head = node
    for i, num in enumerate(seq):
        node.data = num
        if i == len(seq) - 1:
            node.next = None
        else:
            node.next = ListNode()
        node = node.next
    return head

def list_to_seq(L):
    """
    Given a linked list, returns it as a list
    """
    l = []
    while L:
        l.append(L.data)
        L = L.next
    return l

# Sanity checks
A = [1, 2]
assert list_to_seq(seq_to_list(A)) == A


# Linked list classes
class SinglyLinkedList:
    def __init__(self):
        self.head = None
    
    def prepend(self, data):
        """
        Insert a new element at the beginning of the list.
        """
        node = ListNode(data=data)
        node.next = self.head
        self.head = node
    
    def append(self, data):
        """
        Insert a new element at the end of the list.
        """
        if not self.head:
            self.head = ListNode(data=data)
        else:
            curr = self.head
            while curr.next:
                curr = curr.next
            curr.next = ListNode(data=data)
    
    def find(self, key):
        """
        Search for the first element with `data` matching
        `key`. Return the element or `None` if not found.
        """
        curr = self.head
        while curr and curr.data != key:
            curr = curr.next
        return curr
    
    def remove(self, key):
        """
        Remove the first occurrence of `key` in the list.
        """
        curr = self.head
        prev = None
        while curr.data != key:
            prev = curr
            curr = curr.next
        if curr:
            if prev:
                prev.next = curr.next
                curr.next = None
            else:
                self.head = curr.next
    
    def reverse(self):
        """
        Reverse the list in-place
        """
        curr = self.head
        prev, next = None
        while curr:
            next = curr.next
            curr.next = prev
            prev, curr = curr, next
        self.head = prev

        
class DListNode:
    def __init__(self, data=None):
        self.data = data
        self.prev = self.next = None
        
        
class DoublyLinkedList:
    def __init__(self):
        """
        Create a new doubly linked list.
        """
        self.head = None

    def __repr__(self):
        """
        Return a string representation of the list.
        """
        nodes = []
        curr = self.head
        while curr:
            nodes.append(repr(curr))
            curr = curr.next
        return '[' + ', '.join(nodes) + ']'
    
    def prepend(self, data):
        """
        Insert a new element at the beginning of the list.
        """
        node = DListNode(data=data)
        node.next = self.head.next
        node.prev = self.head
        if self.head:
            self.head.prev = node
        self.head = node
    
    def append(self, data):
        """
        Insert a new element at the end of the list.
        """
        if not self.head:
            self.head = DListNode(data=data)
        else:
            curr = self.head
            while curr.next:
                curr = curr.next
            curr.next = DListNode(data=data)
            curr.next.prev = curr
    
    def remove_elem(self, node):
        """
        Unlink an element from the list.
        """
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if node is self.head:
            self.head = node.next
        node.next = node.prev = None
    
    def find(self, key):
        """
        Search for the first element with `data` matching
        `key`. Return the element or `None` if not found.
        """
        curr = self.head
        while curr and curr.data != key:
            curr = curr.next
        return curr

    def remove(self, key):
        """
        Remove the first occurrence of `key` in the list.
        """
        node = self.find(key)
        if node:
            remove_elem(node)
    
    def reverse(self):
        """
        Reverse the list in-place.
        """
        curr = self.head
        prev = None
        while curr:
            curr.prev, curr.next = curr.next, curr.prev
            if curr.prev is None:
                self.head = curr
            curr = curr.prev

def merge(L1, L2):
    """
    Given two sorted linked lists represented by LinkNodes
    Returns the merged linked list
    """
    dummy_head = curr = ListNode()
    while L1 and L2:
        if L1.data < L2.data:
            curr.next = L1
            L1 = L1.next
        else:
            curr.next = L2
            L2 = L2.next
        curr = curr.next
    curr.next = L1 or L2
    return dummy_head.next


# Tests
L1 = seq_to_list([1, 3, 5])
L2 = seq_to_list([2, 4])
assert list_to_seq(merge(L1, L2)) == [1, 2, 3, 4, 5]

def merge_2(L1, L2):
    """
    Given two sorted doubly linked lists
    Merges and returns them
    """
    dummy_head = curr = DListNode()
    while L1 and L2:
        if L1.data < L2.data:
            curr.next, L1.prev = L1, curr
            L1 = L1.next
        else:
            curr.next, L2.prev = L2, curr
            L2 = L2.next
        curr = curr.next
    
    if L1:
        curr.next, L1.prev = L1, curr
    else:
        curr.next, L2.prev = L2, curr
    
    dummy_head.next.prev = None
    return dummy_head.next

def reverse_sublist(L, start, finish):
    """
    Given a linked list
    Reverses the sublist (start, finish)
    """
    dummy_head = sublist_head = ListNode()
    dummy_head.next = sublist_head.next = L
    
    for _ in range(1, start):
        sublist_head = sublist_head.next
    
    sublist_node = sublist_head.next
    for _ in range(finish - start):
        next_node = sublist_node.next
        sublist_node.next = next_node.next
        next_node.next = sublist_head.next
        sublist_head.next = next_node
    return dummy_head.next

# Tests
L1 = seq_to_list([11, 7, 5, 3, 2])
assert list_to_seq(reverse_sublist(L1, 2, 4)) == [11, 3, 5, 7, 2]

def reverse_list(L):
    """
    Reverses the given linked list
    """
    curr = L
    prev_node = next_node = None
    while curr:
        next_node, curr.next = curr.next, prev_node
        prev_node, curr = curr, next_node
    return prev_node

# Tests
L = seq_to_list([1, 2, 3, 4])
assert list_to_seq(reverse_list(L)) == [4, 3, 2, 1]

