def f():
    """Print something to standard out."""
    print('something')

print(dir(f))

g = f
g()

def func_name(function):
    return function.__name__

func_name(f)

function_collection = [f, g]
for function in function_collection:
    function()

def print_when_called(function):
    def new_function(*args, **kwargs):
        print("{} was called".format(function.__name__))
        return function(*args, **kwargs)
    return new_function

def one():
    return 1
one = print_when_called(one)

@print_when_called
def one_():
    return 1

[one(), one_(), one(), one_()]

def print_when_applied(function):
    print("print_when_applied was applied to {}".format(function.__name__))
    return function

@print_when_applied
def never_called():
    import os
    os.system('rm -rf /')

@print_when_applied
@print_when_called
def this_name_will_be_printed_when_called_but_not_at_definition_time():
    pass

this_name_will_be_printed_when_called_but_not_at_definition_time()

@func_name
def a_named_function():
    return

a_named_function

def process_list(list_):
    def decorator(function):
        return function(list_)
    return decorator

unprocessed_list = [0, 1, 2, 3]
special_var = "don't touch me please"

@process_list(unprocessed_list)
def processed_list(items):
    special_var = 1
    return [item for item in items if item > special_var]

(processed_list, special_var)

class FunctionHolder(object):
    def __init__(self, function):
        self.func = function
        self.called_count = 0
    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        finally:
            self.called_count += 1
            
def held(function):
    return FunctionHolder(function)

@held
def i_am_counted():
    pass

i_am_counted()
i_am_counted()
i_am_counted()
i_am_counted.called_count

@FunctionHolder
def i_am_also_counted(val):
    print(val)
    
i_am_also_counted('a')
i_am_also_counted('b')
i_am_also_counted.called_count

@i_am_also_counted
def about_to_be_printed():
    pass

i_am_also_counted.called_count

@len
@func_name
def nineteen_characters():
    """are in this function's name"""
    pass

nineteen_characters

mappings = {'correct': 'good', 'incorrect': 'bad'}

@list
@str.upper
@mappings.get
@func_name
def incorrect():
    pass

incorrect

import re

def constructor(type_):
    def decorator(method):
        method.constructs_type = type_
        return method
    return decorator

def register_constructors(cls):
    for item_name in cls.__dict__:
        item = getattr(cls, item_name)
        if hasattr(item, 'constructs_type'):
            cls.constructors[item.constructs_type] = item
    return cls
            
@register_constructors
class IntStore(object):
    constructors = {}
    
    def __init__(self, value):
        self.value = value
        
    @classmethod
    @constructor(int)
    def from_int(cls, x):
        return cls(x)
    
    @classmethod
    @constructor(float)
    def from_float(cls, x):
        return cls(int(x))
    
    @classmethod
    @constructor(str)
    def from_string(cls, x):
        match = re.search(r'\d+', x)
        if match is None:
            return cls(0)
        return cls(int(match.group()))
    
    @classmethod
    def from_auto(cls, x):
        constructor = cls.constructors[type(x)]
        return constructor(x)
    
IntStore.from_auto('at the 11th hour').value == IntStore.from_auto(11.1).value

def red(fn):
    fn.color = 'red'
    return fn

def blue(fn):
    fn.color = 'blue'
    return fn

@red
def combine(a, b):
    result = []
    result.extend(a)
    result.extend(b)
    return result

@blue
def unsafe_combine(a, b):
    a.extend(b)
    return a

@blue
def combine_and_save(a, b):
    result = a + b
    with open('combined') as f:
        f.write(repr(result))
    return combined

def combine_using(fn, a, b):
    if hasattr(fn, 'color') and fn.color == 'blue':
        print("Sorry, only red functions allowed here!")
        return combine(a, b)  # fall back to default implementation
    return fn(a, b)

a = [1, 2]
b = [3, 4]
print(combine_using(unsafe_combine, a, b))
a

def combine(a, b):
    return a + b
combine.color = 'red'

FUNCTION_REGISTRY = []

def registered(fn):
    FUNCTION_REGISTRY.append(fn)
    return fn

@registered
def step_1():
    print("Hello")
    
@registered
def step_2():
    print("world!")

def run_all():
    for function in FUNCTION_REGISTRY:
        function()
        
run_all()

def step_1():
    print("Hello")
    
def step_2():
    print("world!")
    
FUNCTION_REGISTRY = [step_1, step_2]

def brackets_balanced(s):
    brackets = {
        opening: closing
        for opening, closing in 
        '() {} []'.split()
    }
    closing = set(brackets.values())
    stack = []
    for char in s:
        if char not in closing:
            if char in brackets:
                stack.append(brackets[char])
            continue
        try:
            expected = stack.pop()
        except IndexError:
            return False
        if char != expected:
            return False
    return not stack

def ensure_brackets_balanced(fn):
    for const in fn.__code__.co_consts:
        if not isinstance(const, str) or brackets_balanced(const):
            continue
        print(
            "WARNING - {.__name__} contains unbalanced brackets: {}".format(
                fn, const
            )
        )
    return fn

@ensure_brackets_balanced
def get_root_div_paragraphs(xml_element):
    return xml_element.xpath("//div[not(ancestor::div]/p")

STRATEGIES = []

def precondition(cond):
    def decorator(fn):
        fn.precondition_met = lambda **kwargs: eval(cond, kwargs)
        STRATEGIES.append(fn)
        return fn
    return decorator

@precondition("s.startswith('The year is ')")
def parse_year_from_declaration(s):
    return int(s[-4:])

@precondition("any(substr.isdigit() for substr in s.split())")
def parse_year_from_word(s):
    for substr in s.split():
        try:
            return int(substr)
        except Exception:
            continue
            
@precondition("'-' in s")
def parse_year_from_iso(s):
    from dateutil import parser
    return parser.parse(s).year

def parse_year(s):
    for strategy in STRATEGIES:
        if strategy.precondition_met(s=s):
            return strategy(s)
        
parse_year("It's 2017 bro.")

