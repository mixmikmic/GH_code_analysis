class bundle(object):
    def __init__(self, bun, price, num):
        self.cakes = set(bun)
        self.price = price
        self.n = num
        self.sub = []

    def min_price(self):
        #recursivley find the minimum price
        if self.sub == []:  #base case, there are no children bundles
            return self.price
        else:
            sub_min = sum([b.min_price() for b in self.sub])
            return min(self.price, sub_min)
    
    def is_subset(self, other):
        return self.cakes.issubset(other.cakes)

    def cleanup(self):
        """Remove any subsets that do not have their union equal to 
        their parent bundle"""
        if self.cakes == []:
            return None
        sub_cakes = set().union(*[i.cakes for i in self.sub])
        if self.cakes == sub_cakes:
            for i in self.sub:
                i.cleanup()
        else:
            self.sub = []

    def add(self, bundle):
        """Add the bundle to the smallest bundle 
        of which it is a subset"""
        if self.sub == []:      #case 1, parent bundle has no children
            self.sub.append(bundle)
        else:
            added = False
            for b in self.sub:
                if bundle.is_subset(b): #case 2, child bundle is subset of one of the children
                    b.add(bundle)
                    added = True
                    break
            if not added:       #case 3, child bundle is not a subset of any of the children
                self.sub.append(bundle) 

    def __str__(self):
        return str(self.cakes)


t = int(raw_input().strip())
for a0 in range(t):
    n,m = map(int,raw_input().strip().split())

    bundles = []

    #read in the bundles and create a list of bundle objects
    complete = set()
    for i in range(m):
        b = map(int,raw_input().strip().split())
        bundles.append(bundle(bun = b[2:], price = b[0], num = b[1]))
        complete = complete.union(b[2:])

    #make the complete bundle, and give it a very large price so the min_price function will not return it
    complete_bundle = bundle(complete, 10e9, len(complete))

    #sort by number of cakes in each bundles, most cakes to least
    bundles = sorted(bundles, key=lambda bund:bund.n, reverse=True)

    #create the subset relationship tree, since any two bundles are either 
    #disjoint or have a subset relationship
    for i in bundles:
        complete_bundle.add(i)

    #delete any bundles that are not able to union with its siblings to become the parent bundle
    complete_bundle.cleanup()

    #find the minimum price for getting all bundles
    print complete_bundle.min_price()

