from IPython.display import Image
Image(filename='simpleBST.png', width=300) 

# https://www.laurentluce.com/posts/binary-search-tree-library-in-python/
# The codes bleow are from the above git and web page.

class Node(object):
    """Tree node: left and right child + data which can be any object

    """
    def __init__(self, data):
        """Node constructor
        
        @param data node data object
        """
        self.left = None
        self.right = None
        self.data = data

    def insert(self, data):
        """Insert new node with data

        @param data node data object to insert
        """
        if self.data:
            if data < self.data:
                if self.left is None:
                    self.left = Node(data)
                else:
                    self.left.insert(data) # recursive call
            elif data > self.data:
                if self.right is None:
                    self.right = Node(data)
                else:
                    self.right.insert(data) # recursive call
        else:
            self.data = data

    def lookup(self, data, parent = None):
        """Lookup node containing data

        @param data node data object to look up
        @param parent node's parent
        @returns node and node's parent if found or None, None
        """
        if data < self.data:
            if self.left is None:
                return None, None
            return self.left.lookup(data, self)
        elif data > self.data:
            if self.right is None:
                return None, None
            return self.right.lookup(data, self)
        else: # ==
            return self, parent

    def delete(self, data):
        """Delete node containing data

        @param data node's content to delete
        """
        # get node containing data
        node, parent = self.lookup(data)
        if node is not None:
            children_count = node.children_count()
            if children_count == 0:
                # if node has no children, just remove it
                if parent:
                    if parent.left is node:
                        parent.left = None
                    else:
                        parent.right = None
                else: # why?
                    self.data = None
            elif children_count == 1:
                # if node has 1 child
                # replace node by its child
                if node.left:
                    n = node.left
                else:
                    n = node.right
                if parent:
                    if parent.left is node:
                        parent.left = n
                    else:
                        parent.right = n
                else:
                    self.left = n.left
                    self.right = n.right
                    self.data = n.data
            else:
                # if node has 2 children
                # find its successor
                parent = node
                successor = node.right
                while successor.left:
                    parent = successor
                    successor = successor.left
                # replace node data by its successor data
                node.data = successor.data
                # fix successor's parent node child
                if parent.left == successor:
                    parent.left = successor.right
                else:
                    parent.right = successor.right

    def compare_trees(self, node):
        """Compare 2 trees

        @param node tree to compare
        @returns True if the tree passed is identical to this tree
        """
        if node is None:
            return False
        if self.data != node.data:
            return False
        res = True
        if self.left is None:
            if node.left:
                return False
        else:
            res = self.left.compare_trees(node.left)
        if res is False:
            return False
        if self.right is None:
            if node.right:
                return False
        else:
            res = self.right.compare_trees(node.right)
        return res

    def print_tree(self):
        """Print tree content inorder

        """
        if self.left:
            self.left.print_tree()
        print(self.data, end=" ")
        if self.right:
            self.right.print_tree()

    def children_count(self):
        """Return the number of children

        @returns number of children: 0, 1, 2
        """
        cnt = 0
        if self.left:
            cnt += 1
        if self.right:
            cnt += 1
        return cnt

# __init__

root = Node(8)
Image(filename='bst1.png', width=200) 

# insert()

root.insert(3)
root.insert(10)
root.insert(1)
Image(filename='bst2.png', width=200) 

# insert()

root.insert(6)
root.insert(4)
root.insert(7)
root.insert(14)
root.insert(13)
Image(filename='bst3.png', width=300) 

# lookup()

node, parent = root.lookup(6)

# delete()

root.delete(1)
Image(filename='bst4.png', width=400) 

# delete()

root.delete(14)
Image(filename='bst5.png', width=400) 

# delete()

root.delete(3)
Image(filename='bst6.png', width=400) 

# print_tree()

root.print_tree()

# compare_trees()

root.compare_trees(root.left)

Image(filename='skewedTree.png', width=300) 

# Q: how about delete() and 

Image(filename='unbalanced.png', width=300)

# Again! Computing balance factor in a binary tree can be a 'divide-and-conquer' algorithm.

Image(filename='worstAVL.png', width=600)

# The following trees are still AVL tree, but they are the example of the worst cases of heights 0, 1, 2, and 3.

