import hashlib
import math
import timeit

from IPython.display import display, Markdown

from support.utils import _format_time

algorithms = {'sha512', 'sha1', 'sha256', 'md5', 'sha224', 'sha384'}

def blob(l):
    """return binary blob of length l (POSIX only)"""
    return open("/dev/urandom", "rb").read(l)

def digest(alg, blob):
    md = hashlib.new(alg)
    md.update(blob)
    return md.digest()

def magic_run1(alg, blob):
    t = get_ipython().magic('timeit -o digest(alg, blob)')
    return t

def magic_tfmt(t):
    """format TimeitResult for table"""
    return "{a} ± {s} ([{b}, {w}])".format(
        a = _format_time(t.average),
        s = _format_time(t.stdev),
        b = _format_time(t.best),
        w = _format_time(t.worst),
    )

blob_lengths = [100, 1000, 10000, 100000, 1000000]
blobs = [blob(l) for l in blob_lengths]

table_rows = []
table_rows += [["algorithm"] + list(map(str,blob_lengths))]
table_rows += [["-"] * len(table_rows[0])]
for alg in sorted(algorithms):
    r = [alg]
    for i in range(len(blobs)):
        blob = blobs[i]
        t = timeit.timeit(stmt='digest(alg, blob)', setup='from __main__ import alg, blob, digest', number=1000)
        r += [_format_time(t)]
    table_rows += [r]
table = "\n".join(["|".join(map(str,row)) for row in table_rows])
display(Markdown(table))

def b2B3(b):
    """Convert bits b to Bytes, rounded up modulo 3

    We report modulo 3 because the intent will be to use Base64 encoding, which is
    most efficient when inputs have a byte length modulo 3. (Otherwise, the resulting
    string is padded with characters that provide no information.)
    
    """
    return math.ceil(b/8/3) * 3
    
def B(P, m):
    """return the number of bits needed to achieve a collision probability
    P for m messages

    Assumes m << 2^b.
    
    """
    b = math.log2(m / P) - 1
    if b < 5 + math.log2(m):
        return "-"
    return b2B3(b)

m_bins = [1E6, 1E9, 1E12, 1E15, 1E18, 1E21, 1E24, 1E30]
P_bins = [1E-30, 1E-27, 1E-24, 1E-21, 1E-18, 1E-15, 1E-12, 1E-9, 1E-6, 1E-3, 0.5]

table_rows = []
table_rows += [["#m"] + ["P<={P}".format(P=P) for P in P_bins]]
table_rows += [["-"] * len(table_rows[0])]
for n_m in m_bins:
    table_rows += [["{:g}".format(n_m)] + [B(P, n_m) for P in P_bins]]
table = "\n".join(["|".join(map(str,row)) for row in table_rows])
display(Markdown(table))



