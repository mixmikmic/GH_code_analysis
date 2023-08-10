class TreeNode(object):
    def __init__(self, key, val, left=None, 
                 right=None, parent=None):
        self.key = key
        self.payload = val
        self.leftChild = left
        self.rightChild = right
        self.parent = parent
        
    def hasLeftChild(self):
        return self.leftChild
    
    def hasRightChild(self):
        return self.rightChild
    
    def isLeftChild(self):
        return self.parent and self.parent.leftChild == self
    
    def isRightChild(self):
        return self.parent and self.parent.rightChild == self
    
    def isRoot(self):
        return self.parent == None
    
    def isLeaf(self):
        return not self.leftChild and not self.rightChild
    
    def hasAnyChildren(self):
        return self.leftChild or self.rightChild
    
    def hasBothChildren(self):
        return self.leftChild and self.rightChild
    
    def replaceNodeData(self, key, value, lc, rc):
        self.key = key
        self.payload = value
        self.leftChild = lc
        self.rightChild = rc
        if self.hasLeftChild():
            self.leftChild.parent = self
        if self.hasRightChild():
            self.rightChild.parent = self

class BinarySearchTree(object):
    def __init__(self):
        self.root = None
        self.size = 0
        
    def length(self):
        return self.size
    
    def __len__(self):
        return self.size
    
    def __iter__(self):
        return self.root.__iter__()
    
    def put(self, key, val):
        if self.root:
            self._put(key, val, self.root)
        else:
            self.root = TreeNode(key, val)
        self.size += 1
    
    def _put(self, key, val, currentNode):
        if key < currentNode.key:
            if currentNode.hasLeftChild():
                self._put(key, val, currentNode.leftChild)
            else:
                currentNode.leftChild = TreeNode(key, val, 
                                                 parent=currentNode)
        else:
            if currentNode.hasRightChild():
                self._put(key, val, currentNode.rightChild)
            else:
                currentNode.rightChild = TreeNode(key, val, 
                                                  parent=currentNode)
    
    def __setitem__(self, k, v):
        self.put(k, v)
        
    def get(self, key):
        resNode = self._get(key, self.root)
        if resNode:
            return resNode.payload
        else:
            return None
        
    def _get(self, key, currentNode):
        if currentNode == None:
            return None
        if currentNode.key == key:
            return currentNode
        if key < currentNode.key:
            return self._get(key, currentNode.leftChild)
        else:
            return self._get(key, currentNode.rightChild)
    
    def __getitem__(self, key):
        return self.get(key)
    
    def __contains__(self, key):
        if self._get(key, self.root):
            return True
        else:
            return False
    
    def minimum(self):
        if self.root == None:
            return None
        else:
            minNode = self._minimum(self.root)
            return minNode.key, minNode.payload
        
    def _minimum(self, currentNode):
        if currentNode.leftChild:
            return self._minimum(currentNode.leftChild)
        else:
            return currentNode
        
    def maximum(self):
        if self.root == None:
            return None
        else:
            maxNode = self._maximum(self.root)
            return maxNode.key, maxNode.payload
        
    def _maximum(self, currentNode):
        if currentNode.rightChild:
            return self._maximum(currentNode.rightChild)
        else:
            return currentNode
        
    def successor(self, currentNode):
        if currentNode.hasRightChild():
            return self._minimum(currentNode.rightChild)
        p = currentNode.parent
        while p != None and currentNode == p.rightChild:
            currentNode = p
            p = p.parent
        return p
    
    def predecessor(self, currentNode):
        if currentNode.hasLeftChild():
            return self._maximum(currentNode.leftChild)
        else:
            p = currentNode.parent
            while p != None and currentNode == p.leftChild:
                currentNode = p
                p = p.parent
            return p
    
    def delete(self, key):
        node = self._get(key, self.root)
        if node.isLeaf():
            self.remove(node)
        elif node.leftChild == None or node.rightChild == None:
            self.removeWithOneChild(node)
        elif node.hasBothChildren():
            succeNode = self.successor(node)
            node.key = succeNode.key
            node.payload = succeNode.payload
            if succeNode.isLeaf():
                self.remove(succeNode)
            else:
                self.removeWithOneChild(succeNode)
        
    def remove(self, node):
        if node.isLeftChild():
            node.parent.leftChild = None
        else:
            node.parent.rightChild = None
    
    def removeWithOneChild(self, node):
        if node.hasLeftChild():
            if node.parent == None:
                self.root = node.leftChild
                node.leftChild.parent = None
            else:
                if node.isLeftChild():
                    node.parent.leftChild = node.leftChild
                else:
                    node.parent.rightChild = node.leftChild
                node.leftChild.parent = node.parent
        else:
            if node.parent == None:
                self.root = node.leftChild
                node.leftChild.parent = None
            else:
                if node.isLeftChild():
                    node.parent.leftChild = node.rightChild
                else:
                    node.parent.rightChild = node.rightChild
                node.rightChild.parent = node.parent

b = BinarySearchTree()
l = [17, 5, 35, 2, 29, 33, 38, 16]
for i in l:
    b[i] = i + 1

def preoder(tree):
    currentNode = tree.root
    _preoder(currentNode)

def _preoder(currentNode):
    if currentNode:
        print(currentNode.key, currentNode.payload)
        _preoder(currentNode.leftChild)
        _preoder(currentNode.rightChild)

preoder(b)
b.delete(17)
print("=================")
preoder(b)

print(b.root.key)
print(b.minimum())
print(b.maximum())
print(b.successor(b.root).key)
print(b.predecessor(b.root).key)
print(b.size)
print(b[20],b[21],b[33])
print(32 in b)

