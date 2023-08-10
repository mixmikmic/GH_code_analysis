def gen():
    message = 'Hello'
    while True:
        print('gen: ready to yield %r' % message)
        item = yield message
        print('gen: got %r' % item)
        message = 'message for %s' % item

g = gen()
g

a = next(g)
a

a = g.send('one')
a

a = g.send('two')
a

