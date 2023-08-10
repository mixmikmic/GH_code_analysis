class Fraction:
    def __init__(self, numerator, denominator):
        self.num = numerator
        self.denom = denominator

f = Fraction(4,5)

f.num, f.denom

class Fraction:
    def __init__(self, numerator, denominator):
        self.num = numerator
        self.denom = denominator
    def tofloat(self):
        return float(self.num)/ float(self.denom)

f = Fraction(4,17)
f.tofloat()

def __gcd(a,b):
    while b != 0:
        a, b = b, a%b
    return a

__gcd(4,6), __gcd(55,11), __gcd(3,17)

class Fraction:
    def __init__(self, numerator, denominator):
        self.num = numerator
        self.denom = denominator
    
    def tofloat(self):
        """Return the floating-point value."""
        return float(self.num)/ float(self.denom)
    
    def __gcd(self,a,b):
        while b != 0:
            a, b = b, a%b
        return a
    
    def simplify(self):
        """Simplify the fraction in place."""
        gcd = self.__gcd(self.num, self.denom)
        self.num = int(self.num / gcd)
        self.denom = int(self.denom /gcd)

f = Fraction(15,5)
print( f.num, f.denom, f.tofloat() )
f.simplify()
print( f.num, f.denom, f.tofloat() )

dir(Fraction)

class Fraction:
    def __init__(self, numerator, denominator):
        self.num = numerator
        self.denom = denominator
    
    def tofloat(self):
        """Return the floating-point value."""
        return float(self.num)/ float(self.denom)
    
    def __gcd(self,a,b):
        while b != 0:
            a, b = b, a%b
        return a
    
    def simplify(self):
        """Simplify the fraction in place."""
        gcd = self.__gcd(self.num, self.denom)
        self.num = int(self.num / gcd)
        self.denom = int(self.denom /gcd)
    
    def __str__(self):
        return str(self.num)+"/"+str(self.denom)
    
    def __repr__(self):
        return "Fraction("+str(self.num)+","+str(self.denom)+")"
    
    def __neg__(self):
        return Fraction(-self.num, self.denom)

f = Fraction(4,18)
str(f), repr(f), -f

class Fraction:
    def __init__(self, numerator, denominator):
        self.num = numerator
        self.denom = denominator
    
    def tofloat(self):
        """Return the floating-point value."""
        return float(self.num)/ float(self.denom)
    
    def __gcd(self,a,b):
        while b != 0:
            a, b = b, a%b
        return a
    
    def simplify(self):
        """Simplify the fraction in place."""
        gcd = self.__gcd(self.num, self.denom)
        self.num = int(self.num / gcd)
        self.denom = int(self.denom /gcd)
    
    def __str__(self):
        return str(self.num)+"/"+str(self.denom)
    
    def __repr__(self):
        return "Fraction("+str(self.num)+","+str(self.denom)+")"
    
    def __neg__(self):
        return Fraction(-self.num, self.denom)
    
    def __eq__(self, other):
        a = Fraction(self.num, self.denom)
        a.simplify()
        b = Fraction(other.num, other.denom)
        b.simplify()
        
        return a.num == b.num and a.denom==b.denom
    
    def __gt__(self, other):
        a = Fraction(self.num, self.denom)
        a.simplify()
        b = Fraction(other.num, other.denom)
        b.simplify()
        
        return a.num*b.denom > b.num*a.denom
    
    def __ge__(self, other):
        a = Fraction(self.num, self.denom)
        a.simplify()
        b = Fraction(other.num, other.denom)
        b.simplify()
        
        return a.num*b.denom > b.num*a.denom
    
    def __add__(self, other):
        f = Fraction(self.num*other.denom + other.num*self.denom, self.denom*other.denom)
        f.simplify()
        return f

    def __sub__(self, other):
        f = Fraction(self.num*other.denom - other.num*self.denom, self.denom*other.denom)
        f.simplify()
        return f 
    
    def __mul__(self, other):
        f = Fraction(self.num*other.num, self.denom*other.denom)
        f.simplify()
        return f

    def __truediv__(self, other):
        f = Fraction(self.num*other.denom, self.denom*other.num)
        f.simplify()
        return f

a = Fraction(3,5)
b = Fraction(9,15)
c = Fraction(4, 11)

a==b, a>c, a<c

a+b, a-b, a*c, a/c, c/a, a/b



