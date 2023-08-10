#Page 109, Figure 8.8
def findPayment(loan, r, m):
    """Assumes: loan and r are floats, m an int
       Returns the monthly payment for a mortgage of size
       loan at a monthly rate of r for m months"""
    return loan*((r*(1+r)**m)/((1+r)**m - 1))
# compute the total payment for a $1000 loan whose term is 1 year with a monthly rate of 1%
12*findPayment(1000,0.01,12)

class Mortgage(object):
    """Abstract class for building different kinds of mortgages"""
    def __init__(self, loan, annRate, months):
        """Create a new mortgage"""
        self.loan = loan
        self.rate = annRate/12.0
        self.months = months
        self.paid = [0.0]
        self.owed = [loan]
        self.payment = findPayment(loan, self.rate, months)
        self.legend = None #description of mortgage
    def makePayment(self):
        """Make a payment"""
        self.paid.append(self.payment)
        reduction = self.payment - self.owed[-1]*self.rate
        self.owed.append(self.owed[-1] - reduction)
    def getTotalPaid(self):
        """Return the total amount paid so far"""
        return sum(self.paid)
    def __str__(self):
        return self.legend

#Page 110, Figure 8.9
class Fixed(Mortgage):
    def __init__(self, loan, r, months):
        Mortgage.__init__(self, loan, r, months)
        self.legend = 'Fixed, ' + str(r*100) + '%'
        
class FixedWithPts(Mortgage):
    def __init__(self, loan, r, months, pts):
        Mortgage.__init__(self, loan, r, months)
        self.pts = pts
        self.paid = [loan*(pts/100.0)]
        self.legend = 'Fixed, ' + str(r*100) + '%, '                      + str(pts) + ' points'

#Page 111, Figure 8.10
class TwoRate(Mortgage):
    def __init__(self, loan, r, months, teaserRate, teaserMonths):
        Mortgage.__init__(self, loan, teaserRate, months)
        self.teaserMonths = teaserMonths
        self.teaserRate = teaserRate
        self.nextRate = r/12.0
        self.legend = str(teaserRate*100)                      + '% for ' + str(self.teaserMonths)                      + ' months, then ' + str(r*100) + '%'
    def makePayment(self):
        if len(self.paid) == self.teaserMonths + 1:
            self.rate = self.nextRate
            self.payment = findPayment(self.owed[-1], self.rate,
                                       self.months - self.teaserMonths)
        Mortgage.makePayment(self)

#Page 111, Figure 8.11
def compareMortgages(amt, years, fixedRate, pts, ptsRate,
                     varRate1, varRate2, varMonths):
    totMonths = years*12
    fixed1 = Fixed(amt, fixedRate, totMonths)
    fixed2 = FixedWithPts(amt, ptsRate, totMonths, pts)
    twoRate = TwoRate(amt, varRate2, totMonths, varRate1, varMonths)
    morts = [fixed1, fixed2, twoRate]
    for m in range(totMonths):
        for mort in morts:
            mort.makePayment()
    for m in morts:
        print m
        print ' Total payments = $' + str(int(m.getTotalPaid()))

compareMortgages(amt=200000, years=30, fixedRate=0.07,
                 pts = 3.25, ptsRate=0.05, varRate1=0.045,
                 varRate2=0.095, varMonths=48)

get_ipython().run_cell_magic('writefile', 'compareMortgages.py', '#!/Users/yoavfreund/anaconda/bin/python\ndef findPayment(loan, r, m):\n    """Assumes: loan and r are floats, m an int\n       Returns the monthly payment for a mortgage of size\n       loan at a monthly rate of r for m months"""\n    return loan*((r*(1+r)**m)/((1+r)**m - 1))\n    \nclass Mortgage(object):\n    """Abstract class for building different kinds of mortgages"""\n    def __init__(self, loan, annRate, months):\n        """Create a new mortgage"""\n        self.loan = loan\n        self.rate = annRate/12.0\n        self.months = months\n        self.paid = [0.0]\n        self.owed = [loan]\n        self.payment = findPayment(loan, self.rate, months)\n        self.legend = None #description of mortgage\n    def makePayment(self):\n        """Make a payment"""\n        self.paid.append(self.payment)\n        reduction = self.payment - self.owed[-1]*self.rate\n        self.owed.append(self.owed[-1] - reduction)\n    def getTotalPaid(self):\n        """Return the total amount paid so far"""\n        return sum(self.paid)\n    def __str__(self):\n        return self.legend\n\n#Page 110, Figure 8.9\nclass Fixed(Mortgage):\n    def __init__(self, loan, r, months):\n        Mortgage.__init__(self, loan, r, months)\n        self.legend = \'Fixed, \' + str(r*100) + \'%\'\n        \nclass FixedWithPts(Mortgage):\n    def __init__(self, loan, r, months, pts):\n        Mortgage.__init__(self, loan, r, months)\n        self.pts = pts\n        self.paid = [loan*(pts/100.0)]\n        self.legend = \'Fixed, \' + str(r*100) + \'%, \'\\\n                      + str(pts) + \' points\'\n\n#Page 111, Figure 8.10\nclass TwoRate(Mortgage):\n    def __init__(self, loan, r, months, teaserRate, teaserMonths):\n        Mortgage.__init__(self, loan, teaserRate, months)\n        self.teaserMonths = teaserMonths\n        self.teaserRate = teaserRate\n        self.nextRate = r/12.0\n        self.legend = str(teaserRate*100)\\\n                      + \'% for \' + str(self.teaserMonths)\\\n                      + \' months, then \' + str(r*100) + \'%\'\n    def makePayment(self):\n        if len(self.paid) == self.teaserMonths + 1:\n            self.rate = self.nextRate\n            self.payment = findPayment(self.owed[-1], self.rate,\n                                       self.months - self.teaserMonths)\n        Mortgage.makePayment(self)\n\n#Page 111, Figure 8.11\ndef compareMortgages(amt, years, fixedRate, pts, ptsRate,\n                     varRate1, varRate2, varMonths):\n    totMonths = years*12\n    fixed1 = Fixed(amt, fixedRate, totMonths)\n    fixed2 = FixedWithPts(amt, ptsRate, totMonths, pts)\n    twoRate = TwoRate(amt, varRate2, totMonths, varRate1, varMonths)\n    morts = [fixed1, fixed2, twoRate]\n    for m in range(totMonths):\n        for mort in morts:\n            mort.makePayment()\n    for m in morts:\n        print m\n        print \' Total payments = $\' + str(int(m.getTotalPaid()))\n\nif __name__ == "__main__":\n## add use of argparse\n    compareMortgages(amt=200000, years=30, fixedRate=0.07,\n                 pts = 3.25, ptsRate=0.05, varRate1=0.045,\n                 varRate2=0.095, varMonths=48)')


get_ipython().system('chmod a+x compareMortgages.py')
get_ipython().system('ls -l ')

get_ipython().system('./compareMortgages.py')



