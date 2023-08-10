from functools import partial

def convert(s):
    converters = (int, float)
    
    for converter in converters:
        try:
            value = converter(s)
        except ValueError:
            pass
        else:
            return value
        
    return s

def process_input(s):
    value = convert(s)
    print('%r becomes %r' % (s, value))

def main():
    prompt = 'gimme: '
    while True:
        s = input(prompt)
        if s == 'quit':
            break
        process_input(s)

main()

def main():
    prompt = 'gimme: '
    for s in iter(partial(input, prompt), 'quit'):
        process_input(s)

main()

prompt = 'gimme: '
get_values = (convert(s) for s in iter(partial(input, prompt), 'quit'))
for value in get_values:
    print(value)

