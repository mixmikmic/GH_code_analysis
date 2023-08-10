from flexx import event

class MyObject(event.HasEvents):
        
    @event.connect('foo')
    def on_foo(self, *events):
        print('received the foo event %i times' % len(events))

ob = MyObject()

ob.emit('foo', {})

ob.emit('foo', {}); ob.emit('foo', {})

ob.on_foo()

class MyObject(event.HasEvents):
        
    @event.connect('foo', 'bar')
    def on_foo_or_bar(self, *events):
        for ev in events:
            print('received the %s event' % ev.type)

ob = MyObject()
ob.emit('foo', {}); ob.emit('foo', {}); ob.emit('bar', {})

@ob.connect('foo')
def on_foo(*events):
    print('foo in other handler')

def on_bar(*events):
    print('bar in other handler')

ob.connect(on_bar, 'bar')  # "classic" connect method

ob.emit('foo', {}); ob.emit('bar', {})

class MyObject(event.HasEvents):

    @event.prop
    def foo(self, v=2):
        ''' This is a float indicating some value '''
        return float(v)
    
    @event.connect('foo')
    def on_foo(self, *events):
        print('foo changed to', events[-1].new_value)

ob = MyObject()

ob.foo = 7

print(ob.foo)

ob = MyObject(foo=12)

import time
class MyObject(event.HasEvents):

    @event.readonly
    def bar(self, v):  # no initial value
        ''' This is an int indicating some value '''
        return int(v)
    
    @event.connect('bar')
    def on_bar(self, *events):
        print('bar changed to', events[-1].new_value)
    
    def do_it(self):
        self._set_prop('bar', time.time())
    
ob = MyObject()

ob.do_it()

ob.do_it()

class MyObject(event.HasEvents):

    @event.emitter
    def mouse_down(self, js_event):
        ''' Event emitted when the mouse is pressed down. '''
        return dict(button=js_event['button'])
    
    @event.connect('mouse_down')
    def on_bar(self, *events):
        for ev in events:
            print('detected mouse_down, button', ev.button)

ob = MyObject()

ob.mouse_down({'button': 1})
ob.mouse_down({'button': 2})

class MyObject(event.HasEvents):

    @event.connect('foo:bb')
    def foo_handler1(*events):
        print('foo B')

    @event.connect('foo:cc')
    def foo_handler2(*events):
        print('foo C')
    
    @event.connect('foo:aa')
    def foo_handler3(*events):
        print('foo A')

ob = MyObject()

ob.emit('foo', {})

ob.disconnect('foo:bb')
ob.emit('foo', {})

class Root(event.HasEvents):

    @event.prop
    def children(self, children):
        assert all([isinstance(child, Sub) for child in children])
        return tuple(children)
    
    @event.connect('children', 'children.*.count')
    def update_total_count(self, *events):
        total_count = sum([child.count for child in self.children])
        print('total count is', total_count)

class Sub(event.HasEvents):
    
    @event.prop
    def count(self, count=0):
        return int(count)

root = Root()
sub1, sub2, sub3 = Sub(count=1), Sub(count=2), Sub(count=3)
root.children = sub1, sub2, sub3

sub1.count = 100

root.children = sub2, sub3

sub4 = Sub()
root.children = sub3, sub4

sub4.count = 10

sub1.count = 1000  # no update, sub1 is not part of root's children



