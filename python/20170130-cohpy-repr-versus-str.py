things = (
    1/3,
    .1,
    1+2j,
    [1, 'hello'],
    ('creosote', 3),
    {2: 'world'},
    {'swallow'},
    [1., {"You're": (3, '''tr"'e'''), 1j: 'complexity'}, 17],
)

def show(function, things):
    print(function)
    for thing in things:
        print(function(thing))

show(str, things)
show(repr, things)

def compare(f1, f2, things):
    print(f1, 'versus', f2)
    for thing in things:
        t1 = f1(thing)
        t2 = f2(thing)
        if t1 == t2:
            print('==', t1)
        else:
            print('different')
            print('    1', len(t1), t1)
            print('    2', len(t2), t2)

compare(str, repr, things)

things = (
    """hello""",
    """    """,
    """ """,
    """I'm here.""",
    """Foo said "Hello world!".""",
    """I'm saying "Hello world!".""",
    """\t """,
    """\001""",
    """π""",
)

show(str, things)

show(repr, things)

compare(str, repr, things)

s = 'I\'m saying "Hello world!".'
print(s)

print(str(s))
print(repr(s))

things = (
    b'a \t\nbyt\'"e string',
)

compare(str, repr, things)

things = (
    # string with three '
    "'''",
    """'''""",
    # string with three "
    '"""',
    '''"""''',
    # a string with three ' and three "
    """'''"""    '''"""''',
    """'''\"\"\"""",
    # multiline string
    '''sometext
    type some more text
    type something different''',
)

compare(str, repr, things)

