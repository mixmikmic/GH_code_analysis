class TreeNode(object):
    def __init__(self, key, val, left=None, 
                 right=None, parent=None):
        self.key = key
        self.payload = val
        self.leftChild = left
        self.rightChild = right
        self.parent = parent
        self.balanceFactor = 0
        
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

class AVLTree(object):
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
            if currentNode.left == None:
                currentNode.left = TreeNode(key, val, 
                                            parent=currentNode)
            else:
                self._put(key, val, currentNode.left)
                self.updateBalance(currentNode.left)
        else:
            if currentNode.right == None:
                currentNode.right = TreeNode(key, val, 
                                             parent=currentNode)
            else:
                self._put(key, val, currentNode.right)
                self.updateBalance(currentNode.right)
                
    def updateBalance(currentNode):
        if currentNode.balanceFactor<-1 or currentNode.balanceFactor>1:
            self.rebalance(currentNode)
            return
        
        if currentNode.parent != None:
            if currentNode.isLeftChild:
                currentNode.parent.balanceFactor += 1
            else:
                currentNode.parent.balanceFactor -= 1
            if currentNode.parent.balanceFactor != 0:
                self.updateBalance(currentNode.parent)
        
    def rebalance(self, currentNode):
        if currentNode.balanceFactor > 0:
            if currentNode.leftChild.balanceFacotr < 0:
                self.leftRotate(currentNode.leftChild)
                self.rightRotate(currentNode)
            else:
                self.rightRotate(currentNode)
        else:
            # sym to above
            pass
    
    def rightRotate(self, currentNode):
        A = currentNode
        B = currentNode.leftChild
        C = currentNode.leftChild.rightChild
        if A.isRoot():
            self.root = B
        else:
            if A.isLeftChild:
                A.parent.leftChild = B
            else:
                A.parent.rightChild = B
        B.parent = A.parent
        if C != None:
            A.leftChild = C
            C.parent = A
        else:
            A.leftChild = None
        B.rightChild = A
        A.parent = B
        # update balanceFactor
        
    def leftRotate(self, currentNode):
        pass

