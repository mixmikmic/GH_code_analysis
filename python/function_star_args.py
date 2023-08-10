first, *body, tail = range(10)
print(body)

print(first)

print(tail)

def func(a, *example_args, b, c):
    print(a, example_args, b, c)
    
func(13, 34, 5, 6, 19, b='Donald', c='typing')

func(13, 34, 5, 6, 19, 'Donald', 'typing')

func(13, [34, 5, 6], b='Donald', c='typing')

func(13, range(4), b='Donald', c='typing')

func(13, *range(4), b='Donald', c='typing')



