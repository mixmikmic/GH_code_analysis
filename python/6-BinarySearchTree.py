class TreeNode:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None

class BinarySearchTree:
    def __init__(self, root=None):
        self.root = root
        
    # Find the node with the minimum value    
    def find_min(self):
        if self.root is None:
            return None
        current = self.root
        while current.leftChild is not None:
            current = current.leftChild
        return current
    
    # Find the node with the maximum value
    def find_max(self):
        if self.root is None:
            return None
        current = self.root
        while current.rightChild is not None:
            current = current.rightChild
        return current
    
    # Insert a node with data into the BST
    def insert(self, data):
        node = TreeNode(data)
        if self.root is None:
            self.root = node
        else:
            current = self.root
            while True:
                if data < current.data:
                    if current.leftChild is None:
                        current.leftChild = node
                        return 
                    current = current.leftChild
                else:
                    if current.rightChild is None:
                        current.rightChild = node
                        return 
                    current = current.rightChild
    
    # Delete a node with data from the BST
    # Not implemented yet.
    # This function is a bit tricky; we need to find the node with data first;
    # then based on how many children it has, proceeds with different actions;
    # 0 or 1 child should be easy, while 2 children is not trivial;
    # need to find from its right child the node with smallest value that is
    # bigger than the target's value
    def delete(self, data):
        pass
    
    # Search for the node with data
    def search(self, data):
        current = self.root
        while current is not None:
            if current.data == data:
                return current
            current = current.leftChild if data < current.data else current.rightChild
        return current

bst = BinarySearchTree()
for num in (7, 5, 9, 8, 15, 16, 18, 17):
    bst.insert(num)
max_node = bst.find_max()
min_node = bst.find_min()
print(f"Max node: {max_node.data}")
print(f"Min node: {min_node.data}")
for n in (17, 3, -2, 8, 5):
    if bst.search(n) is not None:
        print(f"{n} found in the binary search tree! :D")
    else:
        print(f"{n} not found in the tree! :(")

