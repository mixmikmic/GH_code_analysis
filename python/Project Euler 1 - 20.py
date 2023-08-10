# import io
# from IPython.nbformat import current

# def execute_notebook(nbfile):
    
#     with io.open(nbfile) as f:
#         nb = current.read(f, 'json')
    
#     ip = get_ipython()
    
#     for cell in nb.worksheets[0].cells:
#         if cell.cell_type != 'code':
#             continue
#         ip.run_cell(cell.input)
#     return

execute_notebook("Euler_Tools.ipynb")


print sum(x for x in xrange(1, 1000) if (x % 3 == 0) or (x % 5 == 0))

def genFibonacci():
    yield 1
    yield 2
    a, b = 1, 2
    while 1:
        a, b = b, a + b
        yield b

result = 0
for x in genFibonacci():
    if x >= 4000000:
        break
    if x % 2 != 0:
        continue
    result += x
    
print result

#
#
## genPrimes is imported from Euler_Tools.ipynb
#
#

num = 600851475143
factors = []

sqrtNum = int(sqrt(num))
for n in genPrimes():
    if num % n == 0:
        factors.append(n)
        if n == num:
            break
        num = num / n
        print n
        sqrtNum = int(sqrt(num))
        
    if n > sqrtNum:
        factors.append(num)
        break
        
print "factors are: {factors}".format(factors=factors)
print "result = {result}".format(result=factors[-1])
    

def reverseNum(nm):
    result = 0
    while nm > 0:
        result = result * 10 + nm % 10
        nm = nm / 10
    return result

def isPalindrome(nm):
    return nm == reverseNum(nm)

@timeit
def findLargestPalindrome():
    num1 = 999
    lesserOfFirstPair = 100
    largest = (0, 0, 0)
    sqrtlargest = 0
    
    for x in xrange(num1,  100, -1):
        if x < sqrtlargest:
            break
        for y in xrange(x, lesserOfFirstPair, -1):
            if isPalindrome(x*y):
                product = x*y
                if product > largest[0]:
                    largest = (product, x, y)
                    sqrtlargest = int(sqrt(product))
                if lesserOfFirstPair < y:
                    lesserOfFirstPair = y
                break
    return largest

largest = findLargestPalindrome()

print largest

# Its the product of all the primes raised to the highest power below 20

print 16 * 9 * 5 * 7 * 11 * 13 * 17 * 19

n = 100
sum1 = n * (n + 1) * (2 * n + 1) / 6
sum2 = pow(n * (n + 1) / 2, 2)

print sum2 - sum1

result = None
for i, x in enumerate(genPrimes()):
    if i >= 10000:
        result = x
        break
print result

data = """
73167176531330624919225119674426574742355349194934
96983520312774506326239578318016984801869478851843
85861560789112949495459501737958331952853208805511
12540698747158523863050715693290963295227443043557
66896648950445244523161731856403098711121722383113
62229893423380308135336276614282806444486645238749
30358907296290491560440772390713810515859307960866
70172427121883998797908792274921901699720888093776
65727333001053367881220235421809751254540594752243
52584907711670556013604839586446706324415722155397
53697817977846174064955149290862569321978468622482
83972241375657056057490261407972968652414535100474
82166370484403199890008895243450658541227588666881
16427171479924442928230863465674813919123162824586
17866458359124566529476545682848912883142607690042
24219022671055626321111109370544217506941658960408
07198403850962455444362981230987879927244284909188
84580156166097919133875499200524063689912560717606
05886116467109405077541002256983155200055935729725
71636269561882670428252483600823257530420752963450
"""

data = data.strip().replace(" ", '').replace("\n", '')

def product(digits):
    ord0 = ord('0')
    digits = [ord(x) - ord0 for x in digits]
    result = reduce(lambda x, y: x*y, digits, 1)
    return result

@timeit
def findMaxProd13():
    result = max([product(data[pos: pos + 13]) for pos in range(len(data) - 12)])
    return result
    
print findMaxProd13()

@timeit
def findTriplet():
    for c in xrange(500, 0, -1):
        for b in xrange(c-1, 0, -1):
            a = 1000 - c - b
            if a > b:
                break
            if (c*c == a*a + b*b):
                return (a, b, c)
            
result = findTriplet()

print result
print reduce(lambda x, y: x*y, result, 1)

@timeit
def findSum():
    result = 0
    for n in itertools.takewhile(lambda x: x < 2000000, genPrimes()):
        result += n

    return result

print findSum()
    

data = """08 02 22 97 38 15 00 40 00 75 04 05 07 78 52 12 50 77 91 08
49 49 99 40 17 81 18 57 60 87 17 40 98 43 69 48 04 56 62 00
81 49 31 73 55 79 14 29 93 71 40 67 53 88 30 03 49 13 36 65
52 70 95 23 04 60 11 42 69 24 68 56 01 32 56 71 37 02 36 91
22 31 16 71 51 67 63 89 41 92 36 54 22 40 40 28 66 33 13 80
24 47 32 60 99 03 45 02 44 75 33 53 78 36 84 20 35 17 12 50
32 98 81 28 64 23 67 10 26 38 40 67 59 54 70 66 18 38 64 70
67 26 20 68 02 62 12 20 95 63 94 39 63 08 40 91 66 49 94 21
24 55 58 05 66 73 99 26 97 17 78 78 96 83 14 88 34 89 63 72
21 36 23 09 75 00 76 44 20 45 35 14 00 61 33 97 34 31 33 95
78 17 53 28 22 75 31 67 15 94 03 80 04 62 16 14 09 53 56 92
16 39 05 42 96 35 31 47 55 58 88 24 00 17 54 24 36 29 85 57
86 56 00 48 35 71 89 07 05 44 44 37 44 60 21 58 51 54 17 58
19 80 81 68 05 94 47 69 28 73 92 13 86 52 17 77 04 89 55 40
04 52 08 83 97 35 99 16 07 97 57 32 16 26 26 79 33 27 98 66
88 36 68 87 57 62 20 72 03 46 33 67 46 55 12 32 63 93 53 69
04 42 16 73 38 25 39 11 24 94 72 18 08 46 29 32 40 62 76 36
20 69 36 41 72 30 23 88 34 62 99 69 82 67 59 85 74 04 36 16
20 73 35 29 78 31 90 01 74 31 49 71 48 86 81 16 23 57 05 54
01 70 54 71 83 51 54 69 16 92 33 48 61 43 52 01 89 19 67 48
"""

data = [[int(x) for x in d.split(" ")] for d in data.strip().split("\n")]

@timeit
def findGreatestProduct():
    l2r = [[data[dy][dx+idx] for idx in xrange(4)] for dy in range(0, len(data)) for dx in range(0, len(data[dy]) - 3)]
    up2down = [[data[dy+idx][dx] for idx in xrange(4)] for dy in range(0, len(data) - 3) for dx in range(0, len(data[dy]))]
    diag1 = [[data[dy+idx][dx+idx] for idx in xrange(4)] for dy in range(0, len(data) -3) for dx in range(0, len(data) - 3)]
    diag2 = [[data[dy-idx][dx+idx] for idx in xrange(4)] for dy in range(3, len(data)) for dx in range(0, len(data) - 3)]
    
    sums = l2r + up2down + diag1 + diag2
    sums = [product(s) for s in sums]
    
    value = max(sums)
    return value
    
print findGreatestProduct()

from itertools import combinations

def genTriangleNums(limit=99999999):
    idx1 = 0
    sidx = 0
    while sidx < limit:
        idx1 += 1
        sidx += idx1
        yield sidx

@timeit
def findTriangleNumber():
    index = 0
    for NUM in genTriangleNums():
        index += 1
        factors = factorize(NUM)
        
        factors1 = [[pow(num, c) for c in range(1, count+1)]for num, count in factors]
        
        divs = []
        for facs in factors1:
            divs += facs

        divisors = set()

        for r in range(1, len(divs) + 1):
            for nums in combinations(divs, r):
                prod = product(nums)
                if NUM % prod == 0:
                    divisors.add(prod)
                    
        numdivs = len(divisors)
        numdivs += 1

        if numdivs > 500:
            break
    value = "%s has %s divisors. This is the %s'th triangle number" % (NUM, numdivs, index)
    return value
    
print findTriangleNumber()

data = """37107287533902102798797998220837590246510135740250
46376937677490009712648124896970078050417018260538
74324986199524741059474233309513058123726617309629
91942213363574161572522430563301811072406154908250
23067588207539346171171980310421047513778063246676
89261670696623633820136378418383684178734361726757
28112879812849979408065481931592621691275889832738
44274228917432520321923589422876796487670272189318
47451445736001306439091167216856844588711603153276
70386486105843025439939619828917593665686757934951
62176457141856560629502157223196586755079324193331
64906352462741904929101432445813822663347944758178
92575867718337217661963751590579239728245598838407
58203565325359399008402633568948830189458628227828
80181199384826282014278194139940567587151170094390
35398664372827112653829987240784473053190104293586
86515506006295864861532075273371959191420517255829
71693888707715466499115593487603532921714970056938
54370070576826684624621495650076471787294438377604
53282654108756828443191190634694037855217779295145
36123272525000296071075082563815656710885258350721
45876576172410976447339110607218265236877223636045
17423706905851860660448207621209813287860733969412
81142660418086830619328460811191061556940512689692
51934325451728388641918047049293215058642563049483
62467221648435076201727918039944693004732956340691
15732444386908125794514089057706229429197107928209
55037687525678773091862540744969844508330393682126
18336384825330154686196124348767681297534375946515
80386287592878490201521685554828717201219257766954
78182833757993103614740356856449095527097864797581
16726320100436897842553539920931837441497806860984
48403098129077791799088218795327364475675590848030
87086987551392711854517078544161852424320693150332
59959406895756536782107074926966537676326235447210
69793950679652694742597709739166693763042633987085
41052684708299085211399427365734116182760315001271
65378607361501080857009149939512557028198746004375
35829035317434717326932123578154982629742552737307
94953759765105305946966067683156574377167401875275
88902802571733229619176668713819931811048770190271
25267680276078003013678680992525463401061632866526
36270218540497705585629946580636237993140746255962
24074486908231174977792365466257246923322810917141
91430288197103288597806669760892938638285025333403
34413065578016127815921815005561868836468420090470
23053081172816430487623791969842487255036638784583
11487696932154902810424020138335124462181441773470
63783299490636259666498587618221225225512486764533
67720186971698544312419572409913959008952310058822
95548255300263520781532296796249481641953868218774
76085327132285723110424803456124867697064507995236
37774242535411291684276865538926205024910326572967
23701913275725675285653248258265463092207058596522
29798860272258331913126375147341994889534765745501
18495701454879288984856827726077713721403798879715
38298203783031473527721580348144513491373226651381
34829543829199918180278916522431027392251122869539
40957953066405232632538044100059654939159879593635
29746152185502371307642255121183693803580388584903
41698116222072977186158236678424689157993532961922
62467957194401269043877107275048102390895523597457
23189706772547915061505504953922979530901129967519
86188088225875314529584099251203829009407770775672
11306739708304724483816533873502340845647058077308
82959174767140363198008187129011875491310547126581
97623331044818386269515456334926366572897563400500
42846280183517070527831839425882145521227251250327
55121603546981200581762165212827652751691296897789
32238195734329339946437501907836945765883352399886
75506164965184775180738168837861091527357929701337
62177842752192623401942399639168044983993173312731
32924185707147349566916674687634660915035914677504
99518671430235219628894890102423325116913619626622
73267460800591547471830798392868535206946944540724
76841822524674417161514036427982273348055556214818
97142617910342598647204516893989422179826088076852
87783646182799346313767754307809363333018982642090
10848802521674670883215120185883543223812876952786
71329612474782464538636993009049310363619763878039
62184073572399794223406235393808339651327408011116
66627891981488087797941876876144230030984490851411
60661826293682836764744779239180335110989069790714
85786944089552990653640447425576083659976645795096
66024396409905389607120198219976047599490197230297
64913982680032973156037120041377903785566085089252
16730939319872750275468906903707539413042652315011
94809377245048795150954100921645863754710598436791
78639167021187492431995700641917969777599028300699
15368713711936614952811305876380278410754449733078
40789923115535562561142322423255033685442488917353
44889911501440648020369068063960672322193204149535
41503128880339536053299340368006977710650566631954
81234880673210146739058568557934581403627822703280
82616570773948327592232845941706525094512325230608
22918802058777319719839450180888072429661980811197
77158542502016545090413245809786882778948721859617
72107838435069186155435662884062257473692284509516
20849603980134001723930671666823555245252804609722
53503534226472524250874054075591789781264330331690"""

@timeit
def findFirst10DigitsOfSum():
    data2 = [int(x) for x in data.split("\n")]
    value = ("%s" % sum(data2))[:10]
    return value
    
print findFirst10DigitsOfSum()

def getChainLength(num, cl):
    if num in cl:
        return cl[num]
    else:
        if num % 2 == 0:
            return 1 + getChainLength(num / 2, cl)
        return 1 + getChainLength(3*num + 1, cl)

@timeit
def findLargestCollatzChain():
    chainLength = {1: 1}
    for x in xrange(1, 1000000):
        cl = getChainLength(x, chainLength)
        chainLength[x] = cl
        
    revcl = [(cl, val) for val, cl in chainLength.items()]
    value = "Longest chain length of %s starting at %s" % max(revcl)
    return value
    
print findLargestCollatzChain()

@timeit
def findNumRoutes():
    # This is equivalent to choosing 20 downs in a path of length 40
    # which has other 20 rights. 40C20
    value = nChooseR(40, 20)
    return value
    
print findNumRoutes()

@timeit
def findSumOfDigits16():
    value = sum(int(x) for x in str(pow(2, 1000)))
    return value
    
print findSumOfDigits16()

@timeit
def findNumLettersUsed():
    result = [num2words(x) for x in range(1, 1001)]
    result = "".join(result).replace(" ", "")
    
    value = len(result)
    return value
    
print findNumLettersUsed()

data = """75
95 64
17 47 82
18 35 87 10
20 04 82 47 65
19 01 23 75 03 34
88 02 77 73 07 63 67
99 65 04 28 06 16 70 92
41 41 26 56 83 40 80 70 33
41 48 72 33 47 32 37 16 94 29
53 71 44 65 25 43 91 52 97 51 14
70 11 33 28 77 73 17 78 39 68 17 57
91 71 52 38 17 14 91 43 58 50 27 29 48
63 66 04 68 89 53 67 30 73 16 69 87 40 31
04 62 98 27 23 09 70 98 73 93 38 53 60 04 23""" # 0

data = [[int(x) for x in d.split(" ")] for d in data.split("\n")]
data.reverse()

@timeit
def findMaxTotal():
    data1 = data
    for idx1, row in enumerate(data[1:]):
        for idx2, x in enumerate(row):
            val = x
            nrow = data[idx1]
            v1, v2 = nrow[idx2], nrow[idx2 + 1]
            val = val + max(v1, v2)
            row[idx2] = val
        
    value = data[-1][0]
    return value
    
print findMaxTotal()

daysNonLeap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
daysLeap = daysNonLeap[:]
daysLeap[1] = 29

@timeit
def getNumFirstSundays():
    sundays = 0
    day = (1 + 365) % 7
    
    for x in range(1901, 2001):
        if x % 4 == 0:
            dpm = daysLeap
        else:
            dpm = daysNonLeap
        
        if x == 2000:
            dpm = dpm[:-1]
            
        for days in dpm:
            day = (day + days) % 7
            if day == 0:
                sundays += 1

    value = sundays    
    return value
    
print getNumFirstSundays()

print sum((int(x) for x in str(factorial(100))))
