things = (
    1/3,
    .1,
    1+2j,
    [1, 'hello'],
    ('creosote', 3),
    {2: 'world'},
    {'swallow'},
    [1., {"You're": (3, '''tr"'e'''), 1j: 'complexity'}, 17],
    """hello""",
    """    """,
    """ """,
    """I'm here.""",
    """Foo said "Hello world!".""",
    """I'm saying "Hello world!".""",
    """\t """,
    """\001""",
    b'a \t\nbyt\'"e string',
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

def show(function, things):
    print(function)
    for thing in things:
        print(function(thing))

show(ascii, things)
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

compare(ascii, repr, things)

things = (
    'π',
    '\u03c0',
    {'π': 3.14159},
    '안녕',
    '安寧',
    'Déjà vu',
)

show(repr, things)

show(ascii, things)

compare(ascii, repr, things)

a = '\u03c0'
b = 'π'
a, b, a == b

a = {'\u03c0': 3.14159}
b = {'π': 3.14159}
a, b, a == b

a = '\uc548\ub155'
b = '안녕'
a, b, a == b

a = '\u5b89\u5be7'
b = '安寧'
a, b, a == b

a = 'D\xe9j\xe0 vu'
b = 'Déjà vu'
a, b, a == b

