"""Factors-and-primes functions.

Find factors or primes of integers, int ranges and int lists
and sets of integers with most factors in a given integer interval

"""

from pprint import pprint

def factorize(n):
    """Calculate all factors of integer n.
        
    Parameters
    ----------
    n : int
        integer to factorize.
    
    Returns
    -------
    list
        A sorted set of integer factors of n.
        
    """
    factors = []
    if isinstance(n, int) and n > 0:
        if n == 1:
            factors.append(n)
            return factors
        else:
            for x in range(1, int(n**0.5)+1):
                if n % x == 0:
                    factors.append(x)
                    factors.append(n//x)
            return sorted(set(factors))
    else:
        print('factorize ONLY computes with one integer argument > 0')


def primes_between(interval_min, interval_max):
    """Find all primes in the interval.
    
    The interval is defined by interval_min and interval_max.
    
    Parameters
    ----------
    interval_min : int
        Start of the integer range.
    interval_max : int
        End of the integer range.
    
    Returns
    -------
    list
        A sorted set of integer primes in original range.
        
    """
    primes = []
    if (isinstance(interval_min, int) and interval_min > 0 and 
       isinstance(interval_max, int) and interval_max > interval_min):
        if interval_min == 1:
            primes = [1]
        for i in range(interval_min, interval_max):
            if len(factorize(i)) == 2:
                primes.append(i)
        return sorted(primes)
    else:
        print('primes_between ONLY computes over the specified range.')

        
def primes_in(integer_list):
    """Calculate all unique prime numbers.
    
    Calculate the prime numbers in a list of integers.

    Parameters
    ----------
    integer_list : list
        A list of integers to test for primality.
        
    Returns
    -------
    list
        A sorted set of integer primes from original list.
    
    """
    primes = []
    try:
        for i in (integer_list):
            if len(factorize(i)) == 2:
                primes.append(i)
        return sorted(set(primes))
    except TypeError:
        print('primes_in ONLY computes over lists of integers.')


def get_ints_with_most_factors(interval_min, interval_max):
    """Finds the integers with the most factors.
    
    Find the integer or integers with the most factors in a given 
    integer range.
    
    The returned result is a list of tuples, where each tuple is:
    [no_with_most_factors (int), no_of_factors (int), 
    factors (int list)].
    
    Parameters
    ----------
    interval_min : int
        Start of the integer range.
    interval_max : int
        End of the integer range.
    
    Returns
    -------
    list
        A list of tuples showing the results.
        
    """
    max_no_of_factors = 1
    all_ints_with_most_factors = []
    
    #: Find the lowest number with most factors between i_min and i_max
    if interval_check(interval_min, interval_max):
        for i in range(interval_min, interval_max):
            factors_of_i = factorize(i)
            no_of_factors = len(factors_of_i) 
            if no_of_factors > max_no_of_factors:
                max_no_of_factors = no_of_factors
                results = (i, max_no_of_factors, factors_of_i,                            primes_in(factors_of_i))
        all_ints_with_most_factors.append(results)
    
        #: Find any larger numbers with an equal number of factors
        for i in range(all_ints_with_most_factors[0][0]+1, interval_max):
            factors_of_i = factorize(i)
            no_of_factors = len(factors_of_i) 
            if no_of_factors == max_no_of_factors:
                results = (i, max_no_of_factors, factors_of_i,                             primes_in(factors_of_i))
                all_ints_with_most_factors.append(results)
        return all_ints_with_most_factors       
    else:
        print_error_msg() 

            
def print_ints_with_most_factors(interval_min, interval_max):
    """Reports integers with most factors in a given integer range.
    
    The results can consist of the following: 
    1.  All the integers with the most factors
    2.  The number of factors
    3.  The actual factors of each of the integers
    4.  Any prime numbers in the list of factors
    
    Parameters
    ----------
    interval_min : int
        Start of the integer range.
    interval_max : int
        End of the integer range.
        
    Returns
    -------
    list
        A list of tuples showing the integers and factors.
        
    """
    if interval_check(interval_min, interval_max):
        print('\nBetween {} and {} the number/s with the most factors:\n'.
           format(interval_min, interval_max))
        for results in (get_ints_with_most_factors(
                            interval_min, interval_max)):
            print('{} ... with the following {} factors:'
                      .format(results[0], results[1]))
            pprint(results[2])
            print('The prime number factors of {} are:'
                      .format(results[0]))
            pprint(results[3])
    else:
        print_error_msg()

        
def interval_check(interval_min, interval_max):
    """Check type and range of integer interval.
    
    Parameters
    ----------
    interval_min : int
        Start of the integer range.
    interval_max : int
        End of the integer range.
        
    Returns
    -------
    None
    
    """
    if (isinstance(interval_min, int) and interval_min > 0 and 
       isinstance(interval_max, int) and interval_max > interval_min):
        return True
    else:
        return False

def print_error_msg():
    """Print invalid integer interval error message.
    
    Returns
    -------
    None
    
    """
    print('ints_with_most_factors ONLY computes over integer intervals where'
            ' interval_min <= int_with_most_factors < interval_max and'
            ' interval_min >= 1')

factorize(1066)

len(primes_between(1, 1066))

primes_1066 = primes_between(1, 1066)
primes_1066_average = sum(primes_1066) / len(primes_1066)
primes_1066_average

primes_1066_ends3 = [x for x in primes_between(1, 1066) if str(x).endswith('3')]
primes_1066_ends3

len(primes_1066_ends3) / len(primes_1066)

print_ints_with_most_factors(1, 1066)

