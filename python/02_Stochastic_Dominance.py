# defaultdict is a Python dictionary that 
# supports initial values for a key
from collections import defaultdict
import numpy as np

def compute_pdf(time_series):
    d = sorted(time_series)
    di = defaultdict(int)
    inc = 1.0/len(d)
    
    for i in range(len(d)):
        di[d[i]] += 1
    
    val  = []
    prob = []
    
    for k in sorted(di.keys()):
        val.append(k)
        prob.append(inc*di[k])

    return val, prob

# And for the CDF
def compute_cdf(prob):
    cdf = [prob[0]]
    for i in range(1, len(prob)):
        cdf.append(prob[i] + cdf[i-1])
    return cdf

data = [1, 2, 4, 6, 8, 9, 1, 4, 5]
val, pdf = compute_pdf(data)
cdf = compute_cdf(pdf)
print pdf
print cdf

# We need to expand our current vectors over all the events
def expand_vector(events, x, y):
    index = 0
    d_mod = []
    for pnt in events:
        if index >= len(x):
            d_mod.append(y[-1])
        elif pnt < x[index]:
            if index == 0:
                d_mod.append(0.0)
            else:
                d_mod.append(y[index-1])
        elif pnt == x[index]:
            d_mod.append(y[index])
        else:
            index += 1
            if index >= len(x):
                d_mod.append(y[-1])
            elif x[index] == pnt:
                d_mod.append(y[index])
            else:
                d_mod.append(y[index-1])
    return d_mod

def check_fosd(d1, d2):
    val1, pdf1 = compute_pdf(d1)
    val2, pdf2 = compute_pdf(d2)
    cdf1, cdf2 = map(compute_cdf, [pdf1, pdf2])
    points = sorted(list(set(val1+val2)))
    d1_mod = map(lambda x: round(x, 5), expand_vector(points, val1, cdf1))
    d2_mod = map(lambda x: round(x, 5), expand_vector(points, val2, cdf2))
    d1_fosd_d2 = all(map(lambda x, y: x<=y, d1_mod, d2_mod))
    d2_fosd_d1 = all(map(lambda x, y: x>=y, d1_mod, d2_mod))
    return d1_fosd_d2, d2_fosd_d1

def check_sosd(d1, d2):
    val1, pdf1 = compute_pdf(d1)
    val2, pdf2 = compute_pdf(d2)
    cdf1, cdf2 = map(compute_cdf, [pdf1, pdf2])
    points = sorted(list(set(val1+val2)))
    d1_mod = map(lambda x: round(x, 5), expand_vector(points, val1, cdf1))
    d2_mod = map(lambda x: round(x, 5), expand_vector(points, val2, cdf2))
    d1_areas = np.cumsum([d1_mod[i]*(points[i+1]-points[i]) for i in range(len(points)-1)])
    d2_areas = np.cumsum([d2_mod[i]*(points[i+1]-points[i]) for i in range(len(points)-1)])
    d1_sosd_d2 = all(map(lambda x, y: x<=y, d1_areas, d2_areas))
    d2_sosd_d1 = all(map(lambda x, y: x>=y, d1_areas, d2_areas))
    return d1_sosd_d2, d2_sosd_d1

d1 = [80, 80, 30, 30, 30, 60, 50, 50, 50, 50]
d2 = [10, 10, 50, 50, 50, 70, 30, 30, 30, 30]
d3 = [20, 80]
d4 = [0, 100]
# D1 FOSD D2
print check_fosd(d1, d2)
print check_sosd(d1, d2)
print check_fosd(d3, d4)
# D3 FOSD D4
print check_sosd(d3, d4)



