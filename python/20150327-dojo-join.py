''.join(['hello', 'gnew', 'world'])

' '.join(['hello', 'gnew', 'world'])

','.join(['hello', 'gnew', 'world'])

', '.join(['hello', 'gnew', 'world'])

' and '.join(['hello', 'gnew', 'world'])

' and '.join(['hello', 'gnew'])

' and '.join(['hello'])

' and '.join([])

def foo():
    yield 'hello'
    yield 'gnew'
    yield 'world'

' and '.join(foo())

