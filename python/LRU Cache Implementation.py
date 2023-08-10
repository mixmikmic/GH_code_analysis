__author__ = "Nagarajan"

get_ipython().run_cell_magic('prun', '', '\nclass LRUCache(object):\n    """\n    Implementation of an LRU cache using a doubly linked list\n    and dict.\n    It has O(1) lookups and is blazingly fast (for python).\n    """\n    def __init__(self, size):\n        """\n        Setup the size and datastore.\n        Instantiate the doubly linked list and nodeMap\n        """\n        self.maxSize = size\n        self.dlist = DoublyLinkedList()\n        self.keyToNodeMap = {}\n        \n    def get(self, key):\n        """\n        Return the data if its in the cache.\n        Else, get it from the datastore, store it in the cache and \n        return the data.\n        """\n        if key in self.keyToNodeMap:\n            # hit\n            node = self.keyToNodeMap[key]\n            self.dlist.moveToFront(node)\n        else:\n            raise KeyError("Key %s not in cache" % key)\n            \n    def add(self, key, value):\n        if self.getCurrSize() >= self.maxSize:\n            oldnode = self.dlist.removeLast()\n            del self.keyToNodeMap[oldnode.key]\n        node = Node(key, value)\n        self.dlist.insertAtFront(node)\n        self.keyToNodeMap[key] = node\n        \n    def remove(self, key):\n        if key in self.keyToNodeMap:\n            node = self.keyToNodeMap[key]\n            self.dlist.remove(node)\n            del self.keyToNodeMap[key]\n        else:\n            raise KeyError("Key %s not in cache" % key)\n        \n    \n    def set(self, key, value):\n        self.db.set(key, value)\n        node = self.keyToNodeMap.get(key, None)\n        if node is not None:\n            node.value = value\n            \n    def getCurrSize(self):\n        return len(self.keyToNodeMap)\n    \n    def updateSize(self, size):\n        currSize = self.getCurrSize()\n        if size < currSize:\n            diff = currSize - size\n            for x in xrange(diff):\n                node = self.dlist.removeLast()\n                del self.keyToNodeMap[node.key]\n        self.maxSize = size\n        \n    def getKeys(self):\n        return self.keyToNodeMap.keys()\n\n\nclass Node(object):\n    def __init__(self, key, value):\n        self.key = key\n        self.value = value\n        self.left = self.right = None\n        \n\nclass DoublyLinkedList(object):\n    def __init__(self):\n        self.head = self.tail = None\n        \n    def removeLast(self):\n        if self.tail is None:\n            raise Exception("Cannot removeLast from empty list")\n        if self.tail == self.head:\n            last = self.tail\n            self.tail = self.head = None\n        else:\n            last = self.tail\n            self.tail = last.left\n            self.tail.right = None\n        return last\n    \n    def insertAtFront(self, node):\n        if self.head is None:\n            self.head = self.tail = node\n        else:\n            node.right = self.head\n            node.left = None\n            self.head.left = node\n            self.head = node\n        return node\n    \n    def moveToFront(self, node):\n        if node is self.head:\n            return node\n        if node is self.tail:\n            last = node.left\n            last.right = None\n            self.tail = last\n        else:\n            left, right = node.left, node.right\n            left.right, right.left = right, left\n            \n        return self.insertAtFront(node)\n\n\nimport unittest    \nimport mock\nimport sys\n\nclass DBMock(object):\n    get = mock.Mock(return_value="get mock")\n    set = mock.Mock(return_value="set mock")\n\nclass TestLRUCache(unittest.TestCase):\n    def setUp(self):\n        self.db = DBMock()\n        self.cache = LRUCache(4, self.db)\n\n        \n    def test_cacheMissesHitDisk(self):\n        result = self.cache.get(\'key1\')\n        self.assertEqual(result, "get mock")\n        \n        callArgs = self.db.get.call_args\n        self.assertEqual(callArgs, mock.call(\'key1\'))\n        \n    def test_dataSetHitsDisk(self):\n        self.cache.set(\'key2\', \'value2\')\n        self.assertEqual(self.cache.getCurrSize(), 0)\n        callArgs = self.db.set.call_args\n        self.assertEqual(callArgs, mock.call(\'key2\', \'value2\'))\n        \n    def test_cache4Items(self):\n        keys = set()\n        for x in xrange(4):\n            key = \'key%s\' % x\n            keys.add(key)\n            self.cache.get(key)\n            \n        self.assertEqual(self.cache.getCurrSize(), 4)\n        self.assertEqual(set(self.cache.getKeys()), keys)\n        \n    def test_cacheModifications(self):\n        keys = set()\n        for x in xrange(20):\n            key = \'key%s\' % x\n            keys.add(key)\n            self.cache.get(key)\n            \n        self.assertEqual(self.cache.getCurrSize(), 4)\n        self.assertEqual(set(self.cache.getKeys()), set([\'key16\', \'key17\', \'key18\', \'key19\']))\n        \n        self.cache.get(\'keya\')\n        self.assertEqual(set(self.cache.getKeys()), set([\'keya\', \'key17\', \'key18\', \'key19\']))\n\n        self.cache.get(\'keyb\')\n        self.assertEqual(set(self.cache.getKeys()), set([\'keya\', \'keyb\', \'key18\', \'key19\']))\n\n        self.cache.get(\'keyc\')\n        self.assertEqual(set(self.cache.getKeys()), set([\'keya\', \'keyb\', \'keyc\', \'key19\']))\n\n        self.cache.get(\'keya\')\n        self.assertEqual(set(self.cache.getKeys()), set([\'keya\', \'keyb\', \'keyc\', \'key19\']))\n\n        self.cache.get(\'keyd\')\n        self.assertEqual(set(self.cache.getKeys()), set([\'keya\', \'keyb\', \'keyc\', \'keyd\']))\n\n        # keyb gets kicked out\n        self.cache.get(\'keye\')\n        self.assertEqual(set(self.cache.getKeys()), set([\'keya\', \'keye\', \'keyc\', \'keyd\']))\n\n        # keyc gets kicked out\n        self.cache.get(\'keyf\')\n        self.assertEqual(set(self.cache.getKeys()), set([\'keya\', \'keye\', \'keyf\', \'keyd\']))\n        \n        # keya gets kicked out\n        self.cache.get(\'keyg\')\n        self.assertEqual(set(self.cache.getKeys()), set([\'keyg\', \'keye\', \'keyf\', \'keyd\']))\n        \n        \n        \nsuite = unittest.TestLoader().loadTestsFromTestCase( TestLRUCache )\nresult = unittest.TextTestRunner(verbosity=1,stream=sys.stderr).run( suite )\n            \n        ')

get_ipython().run_cell_magic('prun', '', '\ndef fib(n):\n    if n <= 2:\n        return 1\n    return fib(n-1) + fib(n-2)\n\nresult = [fib(x) for x in xrange(28)]\nprint result')


