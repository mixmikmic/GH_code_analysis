import math
import sys
def connectionFunc(srclocs,dstlocs,sigma):
    # Corrected normterm (2015/01/16):
    normterm = 1/(math.pow(sigma,2)*2*math.pi)

    i = 0
    out = []
    for srcloc in srclocs:
        j = 0
        for dstloc in dstlocs:
            dist = math.sqrt(math.pow((srcloc[0] - dstloc[0]),2) + math.pow((srcloc[1] - dstloc[1]),2) + math.pow((srcloc[2] - dstloc[2]),2))

            gauss = normterm*math.exp(-0.5*math.pow(dist/sigma,2))

            if gauss > 0.001:
                conn = (i,j,0,gauss)
                out.append(conn)

            j = j + 1
        i = i + 1

    return out

import math
import sys
def connectionFunc(srclocs,dstlocs,sigma):
    # Corrected normterm (2015/01/16):
    normterm = 1/(math.pow(sigma,2)*2*math.pi)
    print 'normterm:',normterm

    i = 0
    out = []
    for srcloc in srclocs:
        j = 0
        for dstloc in dstlocs:
            dist = math.sqrt(math.pow((srcloc[0] - dstloc[0]),2) + math.pow((srcloc[1] - dstloc[1]),2) + math.pow((srcloc[2] - dstloc[2]),2))

            gauss = normterm*math.exp(-0.5*math.pow(dist/sigma,2))

            # Note we need to exceed some proportion of normterm:
            if gauss > 0.7 * normterm:
                conn = (i,j,0,gauss)
                out.append(conn)

            j = j + 1
        i = i + 1

    return out

import math

t = 0
x = 0
y = 0
z = 0
rowLen = 50
srclocs = []
dstlocs = []
while t < 2500:
    x = t % rowLen
    y = math.floor (t / rowLen)
    t += 1
    loc = [x,y,z]
    srclocs.append(loc)
    loc = [x,y,z+1]
    dstlocs.append(loc)

# Play with sigma:
sigma = 11

# Note how I select a single one of srclocs from the array:
o = connectionFunc ([srclocs[850]],dstlocs,sigma)
print len(o)
print o


#PARNAME=R #LOC=1,1
#PARNAME=Am #LOC=1,2
#HASWEIGHT

def connectionFunc(srclocs,dstlocs,R,Am):

  import math

  # Am is 0.2 for STN, 1 for GPe

  i = 0
  out = []
  for srcloc in srclocs:
    j = 0
    for dstloc in dstlocs:
      dist = math.sqrt(math.pow((srcloc[0] - dstloc[0]),2) + math.pow((srcloc[1] - dstloc[1]),2) + math.pow((srcloc[2] - dstloc[2]),2))

      gauss = Am*math.exp(-1*math.pow(dist/R,2))

      # No test suggested in Mandali et al paper
      if gauss > 0.000000001:
        conn = (i,j,0,gauss)
        out.append(conn)

      j = j + 1
    i = i + 1

  return out

import math

t = 0
x = 0
y = 0
z = 0
rowLen = 50
srclocs = []
dstlocs = []
while t < 2500:
    x = t % rowLen
    y = math.floor (t / rowLen)
    t += 1
    loc = [x,y,z]
    srclocs.append(loc)
    loc = [x,y,z+1]
    dstlocs.append(loc)

# Play with sigma:
sigma = 11
Am = 0.2 # 0.2 for STN, 1 for GPe

# Note how I select a single one of srclocs from the array:
o = connectionFunc ([srclocs[850]],dstlocs,sigma,Am)
print len(o)
#print o

