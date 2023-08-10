class SinglyLinkedListNode:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = None

class LinkedList:
    def __init__(self, root_node):
        self.root_node = root_node
    
    def get_val(self, key, curr_node=None):
        if curr_node is None:
            curr_node = self.root_node
        if curr_node.key == key:
            return curr_node.val
        if curr_node.next is None:
            return None
        return self.get_val(key, curr_node.next)
    
    def set_val(self, key, val, curr_node=None):
        if curr_node is None:
            curr_node = self.root_node
        if curr_node.key == key:
            self.val = val
        if curr_node.next is None:
            new_node = SinglyLinkedListNode(key, val)
            curr_node.next = new_node
            return val
        self.set_val(key, val, curr_node.next)
        
    def del_val(self, key, curr_node=None):
        if curr_node is None:
            curr_node = self.root_node
        if curr_node.key == key:
            return None
        if curr_node.next is None:
            return self.root_node
        if curr_node.next.key == key:
            # To delete the next node, we'll just set next to be the node after
            curr_node.next = curr_node.next.next

import numpy as np

class HashTable:
    def __init__(self, size):
        self.size = size
        self.map = np.empty(self.size, dtype=object)
    
    def set_val(self, key, val):
        hash_val = hash(key)
        idx = hash_val % self.size
        if self.map[idx] is None:
            self.map[idx] = LinkedList(SinglyLinkedListNode(key, val))
            return key, val
        linked_list = self.map[idx]
        return linked_list.set_val(key, val)

    def get_val(self, key):
        hash_val = hash(key)
        idx = hash_val % self.size
        if self.map[idx] is None:
            return None
        linked_list = self.map[idx]
        return linked_list.get_val(key)
    
    def del_val(self, key):
        hash_val = hash(key)
        idx = hash_val % self.size
        if self.map[idx] is None:
            return None
        linked_list = self.map[idx]
        linked_list.del_val(key)

import unittest

class TestHashTable(unittest.TestCase):
    def setUp(self):
        self.hash_table = HashTable(5)
        self.hash_table.set_val('cat', 'food')
        self.hash_table.set_val('dog', 'house')
        self.hash_table.set_val('bird', 'cage')
    
    def test_get(self):
        self.assertEqual(self.hash_table.get_val('cat'), 'food')
        self.assertEqual(self.hash_table.get_val('dog'), 'house')
        self.assertEqual(self.hash_table.get_val('bird'), 'cage')

    def test_del(self):
        self.hash_table.del_val('dog')
        self.assertEqual(self.hash_table.get_val('cat'), 'food')
        self.assertIsNone(self.hash_table.get_val('dog'))
        self.assertEqual(self.hash_table.get_val('bird'), 'cage')

if __name__ == '__main__':
    unittest.main(argv=['Ignore first argument'], exit=False)

from collections import Counter
def ransom_note(magazine, ransom):
    magazine_counts = Counter(magazine)
    for word in ransom:
        if word in magazine_counts and magazine_counts[word] > 0:
            magazine_counts[word] -= 1
        else:
            return False
    return True

class TestRansomNote(unittest.TestCase):
    def test_yes(self):
        magazine = 'give me one grand today night'.split(" ")
        ransom = 'give one grand today'.split(" ")
        self.assertTrue(ransom_note(magazine, ransom))

    def test_no(self):
        magazine = 'two times three is not four'.split(" ")
        ransom = 'two times two is four'.split(" ")
        self.assertFalse(ransom_note(magazine, ransom))

if __name__ == '__main__':
    unittest.main(argv=['Ignore first argument'], exit=False)



