def validate_setup(d):
    """
    Validate keys and top-level types of values for game setup dictionary, raising a
      KeyError or TypeError if invalid, otherwise silent.
    Does not distinguish list or tuple types of snakes/ladders pairs.
    Does not check other data constraints, such as snakes or ladder pair values being
      not equal or clashing with another pair's values.
    """
    keys = sorted(d.keys())
    need_keys = sorted(['size', 'snakes', 'ladders', 'diesides', 'name', 'URL'])
    if keys != need_keys:
        raise KeyError("Correct keys not present in setup dictionary")
    if not isinstance(d['size'], int):
        raise TypeError("'size' must be int")
    if not isinstance(d['diesides'], int):
        raise TypeError("'diesides' must be int")
    for kind in ('snakes', 'ladders'):
        if isinstance(d[kind], (list, tuple)):
            try:
                for i1, i2 in d[kind]:
                    # this loop is agnostic about type of sequence
                    if not isinstance(i1+i2, int):
                        raise TypeError("'{}' must be a sequence of pairs of ints".format(kind))
            except ValueError:
                # too many or not enough values to unpack from d['snakes']
                raise TypeError("'{}' must be a sequence of pairs of ints".format(kind))
        else:
            raise TypeError("'{}' must be a sequence of pairs of ints".format(kind))

    
        
def standardize_setup(d):
    # don't standardize if valid
    validate_setup(d)
    r = d.copy() # copy
    r['snakes'] = [tuple(p) for p in d['snakes']]
    r['ladders'] = [tuple(p) for p in d['ladders']]
    return r
    

A = {'size': 16, 
    'snakes': [(11,4), (15,6)],
    'ladders': [(2,10),(8,14)],
    'diesides': 4,
    'name': "Rob's game",
    'URL': ""
        }

B = {'size': 16, 
    'snakes': [(15,6), (11,4)],
    'ladders': ((2,10),(8,14)),
    'diesides': 4,
    'name': "",
    'URL': ""
        }

# Criterion: A represents the same game as B

C = {'size': 16, 
    'snakes':[(11,5), (15,6)],
    'ladders':[(4,10),(8,14)],
    'diesides': 4,
    'name': "Rob's game",
    'URL': ""
        }

# Criterion: A does not represent the same game as C

D = {'size': 16, 
    'snakes': [[11,5], [15,6]], 
    'ladders': [[4,10], [8,14]],
    'diesides': 4,
    'name': "list-only version",
    'URL': ""
        }

# Criterion: D represents the same game as A

E_invalid = {'size': 'h', 
    'snakes': [[11,5], [15,6]], 
    'ladders': [[4,10], [8,14]],
    'diesides': 4}

def compare_SL_setup(d1, d2):
    # only 4 keys we care about
    test1 = d1['size'] == d2['size']
    test2 = d1['diesides'] == d2['diesides']
    test3 = sorted(d1['snakes']) == sorted(d2['snakes'])
    test4 = sorted(d1['ladders']) == sorted(d2['ladders'])
    return test1 and test2 and test3 and test4
    

assert compare_SL_setup(A, B)
assert not compare_SL_setup(A, C)

import json
with open('game_setup1.json', 'w') as fp:
    # auto-closes file at end of 'with' block
    json.dump(A, fp)

with open('game_setup1.json', 'r') as fp:
    d = standardize_setup(json.load(fp))

standardize_setup(d)

