python_is_awesome = True
print(python_is_awesome)

type(python_is_awesome)

i_am_cool = False
print(i_am_cool)

type(i_am_cool)

2 < 3

3.14159 >= 3.15

42 == 42

-1 != 3

(1 < 2) and (1 > 2)

(1 < 2) or (1 > 2)

not (100**3 < 10000)

# Determine if roots of ax^2 + bx + c = 0
# are real, repeated or complex
# Quadratic formula x = (-b \pm \sqrt{b^2 - 4ac})/2a
a = 10
b = -234
c = 1984
discriminant = b**2 - 4*a*c
if discriminant > 0:
    print("Discriminant =", discriminant)
    print("Roots are real and distinct.")
elif discriminant < 0:
    print("Discriminant =", discriminant)
    print("Roots are complex.")
else:
    print("Discriminant =", discriminant)
    print("Roots are real and repeated.")

x = 8
if x > 1:
    print("Hello!")
if x < 2:
    print("Bonjour!")
else:
    print('Ciao!')

def invertible(M):
    '''Determine if M is invertible.
    
    Input:
        M: list of lists representing a 2 by 2 matrix
    Output
        True if M is invertible and False if not
    Example:
        >>> invertible([[1,2],[3,4]])
        True
    '''
    # A matrix M is invertible if and only if it's determinant is not zero
    # Determinant(M) = ad - bc where M = [[a,b],[c,d]]
    determinant = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    if determinant != 0:
        return True
    else:
        return False

invertible([[1,2],[3,4]])

invertible([[1,1],[3,3]])

