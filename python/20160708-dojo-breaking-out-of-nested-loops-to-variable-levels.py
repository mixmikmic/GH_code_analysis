class FirstLevelBreak(Exception): pass
class SecondLevelBreak(Exception): pass
class ThirdLevelBreak(Exception): pass

def show_break(level, i, x_y_z):
    print(
        '    ',
        'Broke with i={} at x,y,z = {} ({} level) '.format(
        i,
        x_y_z,
        level))

def foo(limit, exception):
    print()
    print('foo(limit=%r, exception=%r)' % (limit, exception))
    i = 0
    try:
        for x in range(2):
            try:
                for y in range(2):
                    try:
                        for z in range(2):
                            print('    ', (x,y,z))
                            if (x,y,z) == limit:
                                raise exception
                            i += 1
                    except ThirdLevelBreak:
                        show_break('third', i, (x,y,z))
            except SecondLevelBreak:
                show_break('second', i, (x,y,z))
    except FirstLevelBreak:
        show_break('first', i, (x,y,z))
    print('exiting with i=%s, x,y,z=%s' % (i, (x,y,z)))
    return i, (x,y,z)

foo((0, 0, 0), ThirdLevelBreak)
foo((0, 0, 0), SecondLevelBreak)
foo((0, 0, 0), FirstLevelBreak)
foo((1, 0, 0), ThirdLevelBreak)
foo((1, 0, 0), SecondLevelBreak)
foo((1, 0, 0), FirstLevelBreak)
foo((None, None, None), None)

