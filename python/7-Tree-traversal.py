class TreeNode():
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        
class Tree():
    def __init__(self, root=None):
        self.root = root

# Three forms of DFT
def preorder_traversal(root):
    if root is None:
        return
    print(root.data, end=" ")
    preorder_traversal(root.left)
    preorder_traversal(root.right)
    
def inorder_traversal(root):
    if root is None:
        return
    inorder_traversal(root.left)
    print(root.data, end=" ")
    inorder_traversal(root.right)
    
def postorder_traversal(root):
    if root is None:
        return 
    postorder_traversal(root.left)
    postorder_traversal(root.right)
    print(root.data, end=" ")

from collections import deque
# BFT
def breadth_first_traversal(root):
    if root is None:
        return 
    q = deque()
    q.append(root)
    while len(q) != 0:
        current = q.popleft()
        print(current.data, end=" ")
        if current.left is not None:
            q.append(current.left)
        if current.right is not None:
            q.append(current.right)

#      4
#     / \
#    9   8
#   /   /
#  5   4
#   \
#   10
n1 = TreeNode(4)
n2 = TreeNode(9)
n3 = TreeNode(8)
n4 = TreeNode(5)
n5 = TreeNode(10)
n6 = TreeNode(4)
tree = Tree(n1)
n1.left = n2
n1.right = n3
n2.left = n4
n4.right = n5
n3.left = n6

preorder_traversal(tree.root)
print()
inorder_traversal(tree.root)
print()
postorder_traversal(tree.root)
print()
breadth_first_traversal(tree.root)

