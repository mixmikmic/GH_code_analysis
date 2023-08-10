OPERATORS = '+', '-', '*', '/'


def f_get_number():
    return int(input('Enter an integer: '))


def f_get_operator():
    return input('Enter an operator (+, -, *, /): ')


def f_calculate(number1, operator, number2):
    return number1+number2 if operator == '+'         else number1-number2 if operator == '-'         else number1/number2 if operator == '/'         else number1*number2 if operator == '*'         else None    


def f_main():
    return f_calculate(
        f_get_number(),
        f_get_operator(),
        f_get_number(),
        )


print('The result is: %s' % f_main())

OPERATORS = '+', '-', '*', '/'


def maybe(fnc):
    
    """Turns Exceptions into return values."""
    
    def inner(*args):
        
        for a in args:
            if isinstance(a, Exception):
                return a
        try:
            return fnc(*args)
        except Exception as e:
            return e
        
    return inner


def repeat(fnc, until):
    
    """Repeats a function until its return value meets
    the stop criterion."""
    
    def inner(*args):

        while True:
            result = fnc(*args)
            if until(result):
                return result
            
    return inner


is_int = lambda i: isinstance(i, int)
get_number = lambda: int(input('Enter an integer: '))
safe_get_number = repeat(maybe(get_number), until=is_int)

is_operator = lambda o: o in OPERATORS
get_operator = lambda: input('Enter an operator (+, -, *, /): ')
safe_get_operator = repeat(get_operator, until=is_operator)


calculate = lambda number1, operator, number2:     number1+number2 if operator == '+'     else number1-number2 if operator == '-'     else number1/number2 if operator == '/'     else number1*number2 if operator == '*'     else None    

main = lambda: calculate(
        safe_get_number(),
        safe_get_operator(),
        safe_get_number(),
        )

forever = lambda retval: False
main_loop = repeat(lambda: print(main()), until=forever)

main_loop()

