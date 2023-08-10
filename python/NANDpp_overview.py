import math

# Returns the value of the index variable i in  iteration number k
def index(k):
    r = math.floor(math.sqrt(k+1/4)-1/2)
    return (k-r*(r+1) if k <= (r+1)*(r+1) else (r+1)*(r+2)-k)

[index(k) for k in range(20)]

# Gets as input a NAND program and a table of values of variables. 
# Runs an iteration of the program, updating the values of the variables as it proceeds
def NANDinterpeter(prog,variables):
    for line in prog.split('\n'):
        (var1,_,var2,__,var3) = line.split()
        variables[var1] = 1-variables.get(var2,0)*variables.get(var3,0)
        
# p.s. This code is a bit brittle: sensitive to spaces, comments, and other issues.

# Gets as input a NAND++ program and an input (as an array of 0/1 values) and returns the evaluation of the program on this input
# Works by repeatedly calling NAND
def NANDPPinterpreter(prog,x):
    n = len(x)
    variables = {}
    for i in range(n):
        variables['x_'+str(i)]=x[i]
        variables['validx_'+str(i)]=1
    k=0
    while True:
        NANDprog = prog.replace( '_i','_'+str(index(k)))
        NANDinterpeter(NANDprog,variables)
        if variables.get('loop',0)==0: break
        k += 1
    
    return variables.get('y_0',0) # assume one bit output

parity = r'''tmpa  := seen_i NAND seen_i
tmpb  := x_i NAND tmpa
val   :=  tmpb NAND tmpb
ns   := s   NAND s
y_0  := ns  NAND ns
u    := val NAND s
v    := s   NAND u
w    := val NAND u
s    := v   NAND w
seen_i := zero NAND zero  
stop := validx_i NAND validx_i
loop := stop     NAND stop'''

NANDPPinterpreter(parity,[0,1,1,0,0,1,1])

NANDPPinterpreter(parity,[0,1,1,0,0,1,0,0,1])

def idxincreasing(prog):
    prog = "atstart_0 := zero NAND zero\n"+prog # ensure atstart is array (1,0,0,0,....)
    prog += r'''
tempidx := breadcrumbs_i NAND indexincreasing
notstart := atstart_i NAND atstart_i
indexincreasing := notstart NAND tempidx
breadcrumbs_i := zero NAND zero'''
    return prog    

parityidx = idxincreasing(parity)
print(parityidx)
                          

# Gets as input a NAND++ program and an input (as an array of 0/1 values) and returns the evaluation of the program on this input
# Works by repeatedly calling NAND
def NANDPPinterpreter(prog,x,track = []):
    n = len(x)
    variables = {}
    for i in range(n):
        variables['x_'+str(i)]=x[i]
        variables['validx_'+str(i)]=1
    k=0
    while True:
        NANDprog = prog.replace( '_i','_'+str(index(k)))
        NANDinterpeter(NANDprog,variables)
        if variables.get('loop',0)==0: break
        for v in track: print(v + " = "+ str(variables.get(v,0)))
        k += 1
    
    return variables.get('y_0',0) # assume one bit output

NANDPPinterpreter(parityidx,[0,1,1,0],["indexincreasing"])

import math

# compute the next-step configuration
# Inputs:
# P: NAND++ program in list of 6-tuples representation  (assuming it has an "indexincreasing" variable)
# conf: encoding of configuration as a string using the alphabet "B","E","0","1".
def next_step(P,conf):
    s = len(P) # numer of lines
    t = max([max(tup[0],tup[2],tup[4]) for tup in P])+1 # number of variables
    line_enc_length = math.ceil(math.log(s+1,2)) # num of bits to encode a line
    block_enc_length = t+3+line_enc_length # num of bits to encode a block (without bookends of "E","B")
    LOOP = 3
    INDEXINCREASING = 5
    ACTIVEIDX = block_enc_length -line_enc_length-1 # position of active flag
    FINALIDX =  block_enc_length  -line_enc_length-2 # position of final flag
    
    def getval(var,idx):
        if idx<s: return int(blocks[idx][var])
        return int(active[var])
    
    def setval(var,idx,v):
        nonlocal blocks, i
        if idx<s: blocks[idx][var]=str(v)
        blocks[i][var]=str(v)
    
    blocks = [list(b[1:]) for b in conf.split("E")[:-1]] # list of blocks w/o initial "B" and final "E"
    
    i = [j for j in range(len(blocks))  if blocks[j][ACTIVEIDX]=="1" ][0]
    active = blocks[i]
    
    p = int("".join(active[-line_enc_length:]),2) # current line to be executed
    
    if p==s: return conf # halting configuration
    
    (a,j,b,k,c,l) = P[p] #  6-tuple corresponding to current line#  6-tuple corresponding to current line
    setval(a,j,1-getval(b,k)*getval(c,l))
    
    new_p = p+1
    new_i = i
    if p==s-1: # last line
        new_p = (s if getval(LOOP,0)==0 else 0)
        new_i = (i+1 if getval(INDEXINCREASING,0) else i-1)
        if new_i==len(blocks): # need to add another block and make it final
            blocks[len(blocks)-1][FINALIDX]="0"
            new_final = ["0"]*block_enc_length
            new_final[FINALIDX]="1"
            blocks.append(new_final)
        
        blocks[i][ACTIVEIDX]="0" # turn off "active" flag in old active block
        blocks[i][ACTIVEIDX+1:ACTIVEIDX+1+line_enc_length]=["0"]*line_enc_length # zero out line counter in old active block
        blocks[new_i][ACTIVEIDX]="1" # turn on "active" flag in new active block
    new_p_s = bin(new_p)[2:]
    new_p_s = "0"*(line_enc_length-len(new_p_s))+new_p_s
    blocks[new_i][ACTIVEIDX+1:ACTIVEIDX+1+line_enc_length] = list(new_p_s) # add binary representation of next line in new active block
    
    return "".join(["B"+"".join(block)+"E" for block in blocks]) # return new configuration
    

# return initial configuration of P with input x
def initial_conf(P,x):
    s = len(P) # numer of lines
    t = max([max(tup[0],tup[2],tup[4]) for tup in P])+1 # number of variables
    
    line_enc_length = math.ceil(math.log(s+1,2)) # num of bits to encode a line
    block_enc_length = t+3+line_enc_length # num of bits to encode a block (without bookends of "E","B")
    
     # largest numerical index:
    largest_idx =  max([max((0 if tup[1]==s else tup[1]),(0 if tup[3]==s else tup[3]),(0 if tup[5]==s else tup[5])) for tup in P]) 
    
    ACTIVEIDX = block_enc_length -line_enc_length-1 # position of active flag
    FINALIDX =  block_enc_length  -line_enc_length-2 # position of final flag
    INITIALIDX =  block_enc_length  -line_enc_length-3 # position of initial flag

    num_blocks = max(len(x),largest_idx+1)
    blocks = [["B"]+["0"]*block_enc_length+["E"] for i in range(num_blocks)]
    blocks[0][1+INITIALIDX]="1"
    blocks[0][1+ACTIVEIDX]="1"
    blocks[num_blocks-1][FINALIDX]="1"
    for i in range(len(x)):
        blocks[i][1]=x[i]
        blocks[i][3]="1"
    return "".join(["".join(b) for b in blocks])
     


def is_halting(P,conf):
    newconf = next_step(P,conf)
    return newconf==conf  


def bold(s,justify=0):
    return "\x1b[1m"+s.ljust(justify)+"\x1b[21m"

def underline(s,justify=0):
    return "\x1b[4m"+s.ljust(justify)+"\x1b[24m"

def red(s,justify=0):
    return  "\x1b[31m"+s.ljust(justify)+"\x1b[0m"


def green(s,justify=0):
    return  "\x1b[32m"+s.ljust(justify)+"\x1b[0m"


def blue(s,justify=0):
    return  "\x1b[34m"+s.ljust(justify)+"\x1b[0m"


def print_conf(P,conf):
    s = len(P) # numer of lines
    t = max([max(tup[0],tup[2],tup[4]) for tup in P])+1 # number of variables
    line_enc_length = math.ceil(math.log(s+1,2)) # num of bits to encode a line
    block_enc_length = t+3+line_enc_length # num of bits to encode a block (without bookends of "E","B")
    LOOP = 3
    INDEXINCREASING = 5
    ACTIVEIDX = block_enc_length -line_enc_length-2 # position of active flag
    FINALIDX =  block_enc_length  -line_enc_length-3 # position of final flag
    
    blocks = [list(b[1:]) for b in conf.split("E")[:-1]] 
    printout = "".join([red("B")+ "".join(b[0:t])+
                      blue("".join(b[t:t+3]))+green("".join(b[t+3:t+3+line_enc_length]))+
                      red("E") for b in blocks])
    print(printout)
             

# Parse NAND++ program as sequence of 6-tuples
def parsepp(prog):
    s = len(prog.split('\n'))
    varsidx = { "x":0, "y":1, "validx":2, "loop": 3, "halted":4, "indexincreasing":5 }
    
    def idxsub(var):
        varsplit = var.split('_')
        varid = varsidx.setdefault(varsplit[0],len(varsidx))
        sub = (0 if len(varsplit)<2 else (int(varsplit[1]) if varsplit[1]!='i' else s))
        return [varid,sub]
    
    L = []
    
    for line in prog.split('\n'):
        (var1,_,var2,__,var3) = line.split()
        L.append(idxsub(var1)+idxsub(var2)+idxsub(var3))
    
    return L

L = parsepp(parityidx)
print(L)

initial_conf(L,"1011")

next_step(L,_)

next_step(L,_)

# Evaluate a NAND++ program (represented as a list of tuples)
# By repeatedly applying the "next step" function
def eval_conf(L,x):
    conf = initial_conf(L,x)
    while not is_halting(L,conf):
        print_conf(L,conf)
        conf = next_step(L,conf)
    return int(conf[2])

eval_conf(L,"101")

