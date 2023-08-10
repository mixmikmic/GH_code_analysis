def probB():
    x = int(raw_input())
    for i in range(x):
        cards = []
        y = int(raw_input())
        for j in range(y):
            cards.append(raw_input())
        hand = set(cards)
        numbers = [number[0] for number in hand].append([number[1] for number in hand])
        if len(hand) == len(cards) and len(numbers) >= y:
            print "possible"
        else:
            print "impossible"

def probB():
    x = int(raw_input())
    for i in range(x):
        cards = []
        y = int(raw_input())
        for j in range(y):
            cards.append(raw_input())
        hand = set(cards)
        numbers = [number[0] for number in hand].append([number[1] for number in hand])
        if len(hand) == len(cards) and len(numbers) >= y:
            print "possible"
        else:
            print "impossible"

def probB():
    digits=[]
    tuples=[]
    integers=0
    T=int(raw_input())
    for i in range(T):
        n=int(raw_input())
        for j in range(n):
            p,q=raw_input().strip().split(" ")
            p=int(p)
            q=int(q)
            #tuple part
            if p<q:
                new_tuple=p,q,0
            elif q<p:
                new_tuple=q,p,0
            elif q==p:
                new_tuple=p,q,1
            while new_tuple in tuples:
                new_tuple[2]+=1
            if new_tuple[2]==2:
                print "impossible"
                #return?
            else:
                tuples.append(new_tuple)

            #digit part
            if p not in digits:
                digits.append(p)
            if q not in digits:
                digits.append(q)

            if len(digits)>=n:
                print "impossible"
                #return?
            else: #this doesnt make sense
                print "possible"

def problem_b():
    def combinations1(possibility, hands):
        x = len(possibility)
        card = hands[x]
        possibility1 = copy.copy(possibility)
        possibility2 = copy.copy(possibility)
        possibility1.append(card[0])
        possibility2.append(card[1])
        #print "length " + str(len(possibility1))
        #print "total " + str(total) 
        if len(possibility1) == len(hands):
            return [possibility1, possibility2]
        else:
            next_card = raw_input().split()
            print possibility1, possibility2
            return [combinations(possibility1,hands), combinations(possibility2, hands)]
    def combinations(possibility, hands):
        x = len(possibility)
        card = hands[x]
        possibility1 = copy.copy(possibility)
        possibility2 = copy.copy(possibility)
        possibility1.append(card[0])
        possibility2.append(card[1])
        #print "length " + str(len(possibility1))
        #print "total " + str(total) 
        if len(possibility1) == len(hands):
            print possibility1, possibility2
            return possibility1, possibility2
        else:
            next_card = raw_input().split()
            print possibility1, possibility2
            return combinations(possibility1,hands), combinations(possibility2, hands)
    cases = int(raw_input())
    for k in xrange(cases):
        cards = int(raw_input())
        hands = []
        for j in xrange(cards):
            hands.append(raw_input().split())
        #first_card = raw_input().split()
        empty = []
        oh_yeah = combinations1(empty, hands)
        for i in len(oh_yeah):
            seen = set(oh_yeah[i])
            if len(seen) == len(oh_yeah[i]):
                print "possible"
                break
        print "impossible"

