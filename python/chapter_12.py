class BinaryNode(object):
    
    def __init__(self, key, parent, left=None, right=None):
        self.key = key
        self.parent = parent
        self.left = None
        self.right = None
        
    def __repr__(self):
        return str(self.key)
    
class Tree(object):
    
    def __init__(self):
        self.root = None
        
    def search(self, key):
        return search(self.root, key)
        
    def insert(self, key):
        node = BinaryNode(key, None)
        pointer = self.root
        parent = None
        while pointer:
            parent = pointer
            if pointer.key > node.key:
                pointer = pointer.left
            else:
                pointer = pointer.right
        node.parent = parent
        if parent == None:
            self.root = node
        elif parent.key > node.key:
            parent.left = node
        else:
            parent.right = node
            
    def delete(self, node):
        if not node.left:
            self.transplant(node, node.right)
        elif not node.right:
            self.transplant(node, node.left)
        else:
            head = successor(node)
            if head.parent != node:
                self.transplant(head, head.right)
                head.right = node.right
                head.right.parent = head
            self.transplant(node, head)
            head.left = node.left
            head.left.parent = head
            
    def transplant(self, old, new):
        if not old.parent:
            self.root = new
        elif old == old.parent.left:
            old.parent.left = new
        else:
            old.parent.right = new
        if new:
            new.parent = old.parent
            

def inorder_walk(node):
    result = []
    if node:
        result.extend(inorder_walk(node.left))
        result.append(node.key)
        result.extend(inorder_walk(node.right))
    return result

def preorder_walk(node):
    result = []
    if node:
        result.append(node.key)
        result.extend(preorder_walk(node.left))
        result.extend(preorder_walk(node.right))
    return result

def postorder_walk(node):
    result = []
    if node:
        result.extend(postorder_walk(node.left))
        result.extend(postorder_walk(node.right))
        result.append(node.key)
    return result

def search(node, key):
    while (node) and (node.key != key):
        if node.key > key:
            node = node.left
        else:
            node = node.right
    return node
    
def minimum(node):
    while node.left:
        node = node.left
    return node

def maximum(node):
    while node.right:
        node = node.right
    return node

def successor(node):
    if node.right:
        return minimum(node.right)
    else:
        sucessor = node.parent
        while (successor) and (node == successor.right):
            node = successor
            successor = successor.parent
        return successor
    
def predecessor(node):
    if node.left:
        return maximum(node.left)
    else:
        predecessor = node.parent
        while (successor) and (node == predecessor.left):
            node = predecessor
            predecessor = predecessor.parent
        return predecessor

tree = Tree()
for key in [5, 6, 4, 7, 3, 8, 2, 0, 9, 1]:
    tree.insert(key)
inorder_walk(tree.root)

preorder_walk(tree.root)

postorder_walk(tree.root)

maximum(tree.root)

minimum(tree.root)

successor(tree.root)

predecessor(tree.root)

tree.search(10)

tree.delete(tree.search(5))

preorder_walk(tree.root)



