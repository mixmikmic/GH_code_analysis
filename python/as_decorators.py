get_ipython().magic('xmode plain')

from battle_tested import battle_tested

@battle_tested()
def hardened_int(a):
    """ makes an int no matter what """
    try:
        return int(a)
    except (ValueError, TypeError):
        return 0



