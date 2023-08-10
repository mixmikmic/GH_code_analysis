import numpy as np
import math
from datetime import datetime

get_ipython().magic('load_ext Cython')

2**2-1

2**3-1

2**5-1

2**7-1

2**11-1

p = 74207281
the_number = (2 ** p) - 1

S = 4
print(S)
print(len(str(S)))

S = S ** 2 - 2
print(S)
print(len(str(S)), "digits")

S = S ** 2 - 2
print(S)
print(len(str(S)), "digits")

S = S ** 2 - 2
print(S)
print(len(str(S)), "digits")

S = S ** 2 - 2
print(S)
print(len(str(S)), "digits")

S = S ** 2 - 2
print(S)
print(len(str(S)), "digits")

# The largest prime (so far)
the_number = 2 ** 74207281 - 1
print(int(math.log10(the_number))+1, "digits")

p = 74207281
the_number = (2 ** p) - 1

S = 4
time_stamp = datetime.now()
for i in range(p-2):
    S = (S ** 2 - 2) % the_number
    if i % 1 == 0:
        print(i, datetime.now() - time_stamp,"")
        time_stamp = datetime.now()
if S == 0:
    print("PRIME")

get_ipython().run_cell_magic('cython', '', 'cdef unsigned long p = 61\ncdef unsigned long P = (2 ** p) - 1\n\nS = 4\nfor i in range(p-2):\n    S = S ** 2 - 2\n    S = S % P\n    if i % 10 == 0:\n        print(i)\nif S == 0:\n    print("PRIME")')

get_ipython().run_cell_magic('cython', '--link-args=-lgmp', '\ncdef extern from "gmp.h":\n    ctypedef struct mpz_t:\n        pass\n    \n    ctypedef unsigned long mp_bitcnt_t\n    \n    cdef void mpz_init(mpz_t)\n    cdef void mpz_init_set_ui(mpz_t, unsigned int)\n    \n    cdef void mpz_add(mpz_t, mpz_t, mpz_t)\n    cdef void mpz_add_ui(mpz_t, const mpz_t, unsigned long int)\n    \n    cdef void mpz_sub (mpz_t, const mpz_t, const mpz_t)\n    cdef void mpz_sub_ui (mpz_t, const mpz_t, unsigned long int)\n    cdef void mpz_ui_sub (mpz_t, unsigned long int, const mpz_t)\n    \n    cdef void mpz_mul (mpz_t, const mpz_t, const mpz_t)\n    cdef void mpz_mul_si (mpz_t, const mpz_t, long int)\n    cdef void mpz_mul_ui (mpz_t, const mpz_t, unsigned long int)\n    \n    cdef void mpz_mul_2exp (mpz_t, const mpz_t, mp_bitcnt_t)\n    \n    cdef void mpz_mod (mpz_t, const mpz_t, const mpz_t)\n    \n    cdef unsigned long int mpz_get_ui(const mpz_t)\n\n#cdef unsigned long p = 61\ncdef mp_bitcnt_t p = 74207281\ncdef mpz_t t # = 1\ncdef mpz_t a # = 1\ncdef mpz_t P # = (2 ** p) - 1\ncdef mpz_t S # = 4\n\nmpz_init(t)\nmpz_init_set_ui(t, 1)\n\nmpz_init(a)\nmpz_init_set_ui(a, 2)\n\nmpz_init(P)\nmpz_mul_2exp(P,t,p)\nmpz_sub_ui(P,P,1)\n\nmpz_init(S)\nmpz_init_set_ui(S, 4)\n\nfor i in range(p-2):\n    #S = S ** 2 - 2\n    mpz_mul(S,S,S)\n    mpz_sub_ui(S,S,2)\n    \n    #S = S % P\n    mpz_mod(S,S,P)\n    \n    if i % 1000 == 0:\n        print(i)\nif mpz_get_ui(S) == 0:\n    print("PRIME")\nelse:\n    print("COMPOSITE")\n\n#print(mpz_get_ui(P))')



