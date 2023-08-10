from enum import Enum
import heapq

class Direction(Enum):
    buy = 0
    sell = 1

class Order(object):
    def __init__(self, order_id, quantity, broker, timestamp, direction):
        self.order_id = order_id
        self.quantity = quantity
        self.broker = broker
        self.timestamp = timestamp
        self.direction = direction

    def get_order_id(self):
        return self.order_id

    def get_broker(self):
        return self.broker

    def get_timestamp(self):
        return self.timestamp

    def get_direction(self):
        return self.direction

    def get_quantity(self):
        return self.quantity

class LimitOrder(Order):
    def __init__(self, order_id, quantity, broker, timestamp, direction, price):
        Order.__init__(self, order_id, quantity, broker, timestamp, direction)
        self.price = price

    def signed_price(self):
        if self.direction == Direction.buy.value:
            return +self.price
        else:
            return -self.price

    def __lt__(self, other):
        return (self.signed_price() > other.signed_price())             or (self.signed_price() == other.signed_price()
                   and self.timestamp < other.timestamp)

    def __eq__(self, other):
        return (self.price == other.price)             and (self.timestamp == other.timestamp)

    def __gt__(self, other):
        return not (self.__lt__(other) or self.__eq__(other))

    def __str__(self):
        return "("                "order_id: %s, quantity: %d, "                "broker: %s, timestamp: %d, "                "direction: %d, price: %8.2f"                ")" %                (self.order_id, self.quantity, self.broker, self.timestamp, self.direction, self.price)


    def set_quantity(self, value):
        self.quantity = value

class OrderBook(object):
    def __init__(self):
        self.bids = [] # empty list
        self.asks = [] # empty list
        # you use these lists to create priority queues.
        # you need priority queues to access to the top element in O(1)
        self.order_index = dict()
        # you need a dictionary or a unordered Hash map to access any element in O(1)
        # Based on a key which is the order_id
        # insertion time complexity

    def add(self, lo):
        if lo.direction == Direction.buy.value:
            print("A new bid is added. "
                  "(" \
               "order_id: %s, quantity: %d, " \
               "broker: %s, price: %8.2f" \
               ")" % \
                (lo.order_id, lo.quantity, lo.broker, lo.price)
)
            heapq.heappush(self.bids, lo)
            self.order_index[ lo.order_id ] = lo
        else:
            heapq.heappush(self.asks, lo)
            print("A new ask is added. "
                  "(" \
               "order_id: %s, quantity: %d, " \
               "broker: %s, price: %8.2f" \
               ")" % \
                (lo.order_id, lo.quantity, lo.broker, lo.price))
            self.order_index[ lo.order_id ] = lo
        if self.asks and self.bids:
            self.__match()

    def __match(self):
        best_bid = self.bids[0]
        best_ask = self.asks[0]
        if best_bid.price >= best_ask.price:
            if best_bid.quantity > best_ask.quantity:
                remain = best_bid.quantity - best_ask.quantity
                print('A bid (ID: %s) and an ask (ID: %s) are matched. '
                      'The ask (ID: %s) is filled with the quantity %d.' \
                      % (best_bid.order_id, best_ask.order_id, best_ask.order_id, best_ask.quantity))
                self.bids[0].set_quantity(remain)
                heapq.heappop(self.asks)
                self.__match()

            elif best_bid.quantity < best_ask.quantity:
                remain = best_ask.quantity - best_bid.quantity
                print('A bid (ID: %s) and an ask (ID: %s) are matched. '
                      'The bid (ID: %s) is filled with the quantity %d.' \
                      % (best_bid.order_id, best_ask.order_id, best_bid.order_id, best_bid.quantity))
                self.asks[0].set_quantity(remain)
                heapq.heappop(self.bids)
                self.__match()

            else:
                print('A bid (ID: %s) and an ask (ID: %s) are matched. '
                      'Both are filled with the quantity %d.' \
                      % (best_bid.order_id, best_ask.order_id, best_ask.quantity))
                heapq.heappop(self.bids)
                heapq.heappop(self.asks)

orders = [LimitOrder('1019', 62, 'L', 1, 1, 10702),
LimitOrder('1006', 8, 'R',1, 1, 10665),
LimitOrder('1004', 74, 'C', 2, 0, 9092),
LimitOrder('1012', 92, 'H', 2, 1, 9684),
LimitOrder('1005', 83, 'X', 4, 1, 9841),
LimitOrder('1017', 89, 'D', 5, 1, 9784),
LimitOrder('1001', 40, 'R', 5, 0, 9521),
LimitOrder('1007', 19, 'N', 5, 1, 10388),
LimitOrder('1013', 97, 'J', 6, 0, 9147),
LimitOrder('1010', 41, 'R', 7, 0, 10572),
LimitOrder('1015', 94, 'G', 8, 0, 10077),
LimitOrder('1003', 91, 'Q', 8, 1, 8695),
LimitOrder('1009', 59, 'S', 10, 0, 11066),
LimitOrder('1018', 68, 'F', 12, 0, 8225),
LimitOrder('1008', 3, 'K', 12, 1, 9849),
LimitOrder('1016', 4, 'D', 13, 0, 8726),
LimitOrder('1002', 83, 'O', 16, 0, 10876),
LimitOrder('1011', 38, 'D', 18, 0, 11142),
LimitOrder('1014', 54, 'Q', 19, 1, 9442),
LimitOrder('1000', 68, 'S', 20, 1, 9287)]

book = OrderBook()

index = 0
for order in orders:
    print('---------------------------')
    print('Event [' + str(index) + ']: ')
    book.add( order )
    print('---------------------------')
    index += 1

    print('Bid (Buy-side) priority queue:')
    if ( book.bids ):
        for i in book.bids:
            print(i)
    else:
        print('Bid is empty.')

    print('Ask (Sell-side) priority queue:')
    if ( book.asks ):
        for i in book.asks:
            print(i)
    else:
        print('Ask is empty.')

    print('\n')

