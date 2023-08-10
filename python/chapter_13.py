class Node(object):
    
    def __init__(self, key):
        self.key = key
        self.parent = None
        self.color = "black"
        self.left = None
        self.right = None
        
class Tree(object):
    
    def __init__(self):
        self.nil = Node(None)
        self.root = self.nil
        
    def minimum(self):
        node = self.root
        while node.left != self.nil:
            node = node.left
        return node

    def maximum(self):
        node = self.root
        while node.right != self.nil:
            node = node.right
        return node

    def search(self, key):
        node = self.root
        while node != self.nil:
            if key == node.key:
                return node
            elif key > node.key:
                node = node.right
            else:
                node = node.left
        return None
    
    def insert(self, key):
        node = Node(key)
        pointer = self.root
        parent = self.nil
        while pointer != self.nil:
            parent = pointer
            if pointer.key > node.key:
                pointer = pointer.left
            else:
                pointer = pointer.right
        node.parent = parent
        if parent == self.nil:
            self.root = node
        elif parent.key > node.key:
            parent.left = node
        else:
            parent.right = node
        node.color = "red"
        node.left = self.nil
        node.right = self.nil
        self.insert_fixup(node)
    
    def delete(self, node):
        pointer = node
        original_color = pointer.color
        if node.left == self.nil:
            child = node.right
            self.transplant(node, node.right)
        elif node.right == self.nil:
            child = node.left
            self.transplant(node, node.left)
        else:
            pointer = successor(node)
            original_color = pointer.color
            child = pointer.right
            if pointer.parent == node:
                child.parent = pointer
            else:
                self.transplant(pointer, pointer.right)
                pointer.right = node.right
                pointer.right.parent = pointer
            self.transplant(node, pointer)
            pointer.left = node.left
            pointer.left.parent = pointer
            pointer.color = node.color
        if original_color == "black":
            self.delete_fixup(child)
    
    def delete_fixup(self, node):
        while (node != self.root) and (node.color == "black"):
            if node == node.parent.left:
                sibling = node.parent.right
                if sibling.color == "red":
                    sibling.color = "black"
                    node.parent.color = "red"
                    self.left_rotate(node.parent)
                    sibling = node.parent.right
                if (sibling.left.color == "black") and (sibling.right.color == "black"):
                    sibling.color = "red"
                    node = node.parent
                else:
                    if sibling.right.color == "black":
                        sibling.left.color = "black"
                        sibling.color = "red"
                        self.right_rotate(sibling)
                        sibling = node.parent.right
                    sibling.color = node.parent.color
                    node.parent.color = "black"
                    sibling.right.color = "black"
                    self.left_rotate(node.parent)
                    node = self.root
            else:
                sibling = node.parent.left
                if sibling.color == "red":
                    sibling.color = "black"
                    node.parent.color = "red"
                    self.left_rotate(node.parent)
                    sibling = node.parent.left
                if (sibling.right.color == "black") and (sibling.left.color == "black"):
                    sibling.color = "red"
                    node = node.parent
                else:
                    if sibling.left.color == "black":
                        sibling.right.color = "black"
                        sibling.color = "red"
                        self.right_rotate(sibling)
                        sibling = node.parent.left
                    sibling.color = node.parent.color
                    node.parent.color = "black"
                    sibling.left.color = "black"
                    self.left_rotate(node.parent)
                    node = self.root
        node.color = "black"
    
    def insert_fixup(self, node):
        while node.parent.color == "red":
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.color == "red":
                    node.parent.color = "black"
                    uncle.color = "black"
                    node.parent.parent.color = "red"
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self.left_rotate(node)
                    node.parent.color = "black"
                    node.parent.parent.color = "red"
                    self.right_rotate(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle.color == "red":
                    node.parent.color = "black"
                    uncle.color = "black"
                    node.parent.parent.color = "red"
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self.right_rotate(node)
                    node.parent.color = "black"
                    node.parent.parent.color = "red"
                    self.left_rotate(node.parent.parent)
        self.root.color = "black"
    
    def left_rotate(self, head):
        child = head.right
        head.right = child.left
        if child.left != self.nil:
            child.left.parent = head
        child.parent = head.parent
        if head.parent == self.nil:
            self.root = child
        elif head == head.parent.left:
            head.parent.left = child
        else:
            head.parent.right = child
        child.left = head
        head.parent = child
    
    def right_rotate(self, head):
        child = head.left
        head.left = child.right
        if child.right != self.nil:
            child.right.parent = head
        child.parent = head.parent
        if head.parent == self.nil:
            self.root = child
        elif head == head.parent.left:
            head.parent.left = child
        else:
            head.parent.right = child
        child.right = head
        head.parent = child
        
    def transplant(self, old, new):
        if old.parent == self.nil:
            self.root = new
        elif old == old.parent.left:
            old.parent.left = new
        else:
            old.parent.right = new
        new.parent = old.parent

def inorder_walk(node):
    result = []
    if node.key:
        result.extend(inorder_walk(node.left))
        result.append(node.key)
        result.extend(inorder_walk(node.right))
    return result

def successor(node):
    if node.right:
        return minimum(node.right)
    else:
        successor = node.parent
        while (successor) and (successor.right == node):
            node = successor
            successor = successor.parent
        return successor
            
def predecessor(node):
    if node.left:
        return maximum(node.left)
    else:
        predecessor = node.parent
        while (predecessor) and (node == predecessor.left):
            node = predecessor
            predecessor = predecessor.parent
        return predecessor

tree = Tree()
for key in [5, 6, 4, 7, 3, 8, 2, 0, 9, 1]:
    tree.insert(key)
inorder_walk(tree.root)

tree.maximum().key

tree.minimum().key

tree.search(9)

tree.search(10)

tree.delete(tree.search(5))



