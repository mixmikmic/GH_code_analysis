class BinaryTree(object):
    def __init__(self,rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self,newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self,newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t


    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self,obj):
        self.key = obj

    def getRootVal(self):
        return self.key

book = BinaryTree('book')
# load up chapter 1 on left
book.insertLeft('Chapter 1')
book.getLeftChild().insertLeft("Section 1.1")
s12 = BinaryTree('Section 1.2')
s12.insertLeft("Section 1.2.1")
s12.insertRight("Section 1.2.2")
book.getLeftChild().rightChild = s12
# load up chapter 2 on right
book.insertRight("Chapter 2")
book.getRightChild().insertLeft("Section 2.1")
s22 = BinaryTree('Section 2.2')
s22.insertLeft("Section 2.2.1")
s22.insertRight("Section 2.2.2")
book.getRightChild().rightChild = s22

def preorder(tree):
    if tree:
        print tree.getRootVal()
        preorder(tree.getLeftChild())
        preorder(tree.getRightChild())

preorder(book)

def postorder(tree):
    if tree != None:
        postorder(tree.getLeftChild())
        postorder(tree.getRightChild())
        print tree.getRootVal()

postorder(book)

def inorder(tree):
    if tree != None:
        inorder(tree.getLeftChild())
        print tree.getRootVal()
        inorder(tree.getRightChild())

inorder(book)



