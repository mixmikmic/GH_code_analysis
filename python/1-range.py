# help(range)

test_cases = (
    (-1,),
    (0,),
    (1,),
    (3,),
    (0, 3),
    (-2, 1),
    (0, 0),
    (0, -1),
    (2, 6, 2),
    (2, 7, 2),
    (2, -1, -2),
    (2, -2, -2),
)

def test():
    for args in test_cases:
        assert tuple(range(*args)) == tuple(grange(*args)), (
            args, tuple(range(*args)), tuple(grange(*args)))

def grange(*args):
    start = 0
    step = 1
    
    try:
        stop, = args
    except ValueError:
        try:
            start, stop = args
        except ValueError:
            try:
                start, stop, step = args
            except ValueError as e:
                raise Exception(e, 'wrong number of arguments')
    assert step != 0, 'arg 3 must not be zero'

    i = start
    if step > 0:
        while i < stop:
            yield i
            i += step
    else:
        while i > stop:
            yield i
            i += step

test()

tuple(grange(3))

def grange(*args):
    def complain_about_wrong_number_of_args(*args):
        raise Exception(ValueError, 'wrong number of arguments')

    def range_args_from_one_arg(*args):
        start = 0
        stop, = args
        step = 1
        return start, stop, step

    def range_args_from_two_args(*args):
        start, stop, = args
        step = 1
        return start, stop, step

    def range_args_from_three_args(*args):
        return args

    arg_parsers = {
        1: range_args_from_one_arg,
        2: range_args_from_two_args,
        3: range_args_from_three_args,
    }

    arg_parser = arg_parsers.get(len(args), complain_about_wrong_number_of_args)
    start, stop, step = arg_parser(*args)

    assert step != 0, 'arg 3 must not be zero'

    i = start
    if step > 0:
        while i < stop:
            yield i
            i += step
    else:
        while i > stop:
            yield i
            i += step

test()

def grange(*args):
    start = 0
    step = 1

    if len(args) == 1:
        stop, = args
    elif len(args) == 2:
        start, stop = args
    elif len(args) == 3:
        start, stop, step = args
    else:
        raise Exception(ValueError, 'wrong number of arguments')
    assert step != 0, 'arg 3 must not be zero'

    i = start
    if step > 0:
        while i < stop:
            yield i
            i += step
    else:
        while i > stop:
            yield i
            i += step

test()

from operator import lt, gt

def grange(*args):
    start = 0
    step = 1

    if len(args) == 1:
        stop, = args
    elif len(args) == 2:
        start, stop = args
    elif len(args) == 3:
        start, stop, step = args
    else:
        raise Exception(ValueError, 'wrong number of arguments')
    assert step != 0, 'arg 3 must not be zero'

    i = start
    compare = lt if step > 0 else gt
    while compare(i, stop):
        yield i
        i += step

test()

