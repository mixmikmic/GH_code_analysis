class Node:
    def __init__(self, next_ = None):
        self.next = next_

class LinkedList:
    def __init__(self):
        self.head = Node()
    
    def append(self, node):
        curr_node = self.head
        while curr_node.next is not None:
            curr_node = curr_node.next
        curr_node.next = node

def has_cycle(head_node):
    walker = head_node
    runner = head_node.next
    while runner != walker:
        if (runner is None) or (runner.next is None):
            return False
        walker = walker.next
        runner = runner.next.next
    return True

import unittest

class TestLinkedList(unittest.TestCase):
    def setUp(self):
        self.ll = LinkedList()
        for _ in range(4):
            self.ll.append(Node())
        self.cycle_ll = LinkedList()
        for _ in range(4):
            self.cycle_ll.append(Node())
        self.cycle_ll.append(cycle_ll.head.next.next)
    
    def test_has_no_cycle(self):
        self.assertFalse(has_cycle(self.ll.head))
    
    def test_has_cycle(self):
        self.assertTrue(has_cycle(self.cycle_ll.head))

if __name__ == '__main__':
    unittest.main(argv=['Ignore first argument'], exit=False)



