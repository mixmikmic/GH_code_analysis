get_ipython().magic('xmode plain')

def list_of_strings_v1(iterable):
    """ converts the iterable input into a list of strings """
    # build the output
    out = [str(i) for i in iterable]
    # validate the output
    for i in out:
        assert type(i) == str
    # return
    return out

list_of_strings_v1(range(10))

from battle_tested import fuzz

fuzz(list_of_strings_v1)

def list_of_strings_v2(iterable):
    """ converts the iterable input into a list of strings """
    try:
        iter(iterable)
        # build the output
        out = [str(i) for i in iterable]
    except TypeError: # raised when input was not iterable
        out = [str(iterable)]
    # validate the output
    for i in out:
        assert type(i) == str
    # return
    return out

fuzz(list_of_strings_v2)

