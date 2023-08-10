states = ['b','y','n','e']

transitions = {('b','y') : 0.2,
               ('b','n') : 0.8,
               ('y','y') : 0.7,
               ('y','n') : 0.2,
               ('y','e') : 0.1,
               ('n','n') : 0.8,
               ('n','y') : 0.1,
               ('n','e') : 0.1
    }

emissions = {'y' : {'A':0.1, 'C':0.4, 'G':0.4, 'T':0.1},
             'n' : {'A':0.25, 'C':0.25, 'G':0.25, 'T':0.25}
    }

sequence = 'ATGCG'

def initialize_matrix(dim1,dim2,value=0):
    F = []
    for i in range(0,dim1):
        F.append([])
        for j in range(0,dim2):
            F[i].append(value)
    return F

def print_matrix(matrix,axis1,axis2):
    w = '{:<10}'
    print w.format('') + w.format('0') + ''.join([w.format(char) for char in axis2]) + w.format('0')
    for i, row in enumerate(matrix):
        print w.format(axis1[i]) + ''.join(['{:<10.2e}'.format(item) for item in row])
        
def print_matrix_p(matrix,axis1,axis2):
    w = '{:<10}'
    print w.format('') + w.format('0') + ''.join([w.format(char) for char in axis2]) + w.format('0')
    for i, row in enumerate(matrix):
        print w.format(axis1[i]) + ''.join(['{:<10s}'.format(item) for item in row])

F = initialize_matrix(len(states),len(sequence)+2)
F[0][0] = 1

print_matrix(F,states,sequence)

for i in range(1,len(states)-1):
        F[i][1] = transitions[(states[0],states[i])]*emissions[states[i]][sequence[0]]
        
print_matrix(F,states,sequence)

for j in range(2,len(sequence)+1): # loops on the symbols
        for i in range(1,len(states)-1): # loops on the states
            p_sum = 0
            for k in range(1,len(states)-1): # loops on all of the possible previous states
                p_sum += F[k][j-1]*transitions[(states[k],states[i])]*emissions[states[i]][sequence[j-1]]
            F[i][j] = p_sum
            
print_matrix(F,states,sequence)

p_sum = 0
for k in range(1,len(states)-1):
    p_sum += F[k][len(sequence)]*transitions[(states[k],states[-1])]
F[-1][-1] = p_sum

print_matrix(F,states,sequence)

def forward(states,transitions,emissions,sequence):
    F = initialize_matrix(len(states),len(sequence)+2)
    F[0][0] = 1
    for i in range(1,len(states)-1):
        F[i][1] = transitions[(states[0],states[i])]*emissions[states[i]][sequence[0]]
    for j in range(2,len(sequence)+1):
        for i in range(1,len(states)-1):
            p_sum = 0
            for k in range(1,len(states)-1):
                p_sum += F[k][j-1]*transitions[(states[k],states[i])]*emissions[states[i]][sequence[j-1]]
            F[i][j] = p_sum
    p_sum = 0
    for k in range(1,len(states)-1):
        p_sum += F[k][len(sequence)]*transitions[(states[k],states[-1])]
    F[-1][-1] = p_sum
    return F

F = initialize_matrix(len(states),len(sequence)+2)
F[-1][-1] = 1

print_matrix(F,states,sequence)

for i in range(1,len(states)-1):
    F[i][-2] = transitions[(states[i],states[-1])]
    
print_matrix(F,states,sequence)

for j in range(len(sequence)-1,0,-1): # loops on the symbols
    for i in range(1,len(states)-1): # loops on the states
        p_sum = 0
        for k in range(1,len(states)-1): # loops on all of the possible successive states
            p_sum += F[k][j+1]*transitions[(states[i],states[k])]*emissions[states[k]][sequence[j]]
        F[i][j] = p_sum

print_matrix(F,states,sequence)

p_sum = 0
for k in range(1,len(states)-1):
    p_sum += F[k][1]*transitions[(states[0],states[k])]*emissions[states[k]][sequence[0]]
F[0][0] = p_sum

print_matrix(F,states,sequence)

def backward(states,transitions,emissions,sequence):
    F = initialize_matrix(len(states),len(sequence)+2)
    F[-1][-1] = 1
    for i in range(1,len(states)-1):
        F[i][-2] = transitions[(states[i],states[-1])]
    for j in range(len(sequence)-1,0,-1): 
        for i in range(1,len(states)-1):
            p_sum = 0
            for k in range(1,len(states)-1):
                p_sum += F[k][j+1]*transitions[(states[i],states[k])]*emissions[states[k]][sequence[j]]
            F[i][j] = p_sum
    p_sum = 0
    for k in range(1,len(states)-1):
        p_sum += F[k][1]*transitions[(states[0],states[k])]*emissions[states[k]][sequence[0]]
    F[0][0] = p_sum
    return F

F = initialize_matrix(len(states),len(sequence)+2)
FP = initialize_matrix(len(states),len(sequence)+2,states[0])
F[0][0] = 1

print_matrix(F,states,sequence)
print
print_matrix_p(FP,states,sequence)

for i in range(1,len(states)-1):
    F[i][1] = transitions[(states[0],states[i])]*emissions[states[i]][sequence[0]]
    
print_matrix(F,states,sequence)

def get_max_val_ind(values):
    max_val = values[0]
    max_ind = 0
    for ind, val in enumerate(values):
        if val>max_val:
            max_val = val
            max_ind = ind
    return max_val, max_ind

for j in range(2,len(sequence)+1): # loops on the symbols
        for i in range(1,len(states)-1): # loops on the states
            values = []
            for k in range(1,len(states)-1): # loops on all of the possible previous states
                values.append(F[k][j-1]*transitions[(states[k],states[i])]*emissions[states[i]][sequence[j-1]]) # appends the value to a list
            max_val, max_ind = get_max_val_ind(values) # finds the maximum and the index of the maximum in the list
            F[i][j] = max_val # sets the probability to the maximum probability
            FP[i][j] = states[max_ind+1] # sets the corresponding pointer to the appropriate previous state

print_matrix(F,states,sequence)
print
print_matrix_p(FP,states,sequence)

values = []
for k in range(1,len(states)-1):
    values.append(F[k][len(sequence)]*transitions[(states[k],states[-1])])
max_val, max_ind = get_max_val_ind(values)
F[-1][-1] = max_val
FP[-1][-1] = states[max_ind+1]

print_matrix(F,states,sequence)
print
print_matrix_p(FP,states,sequence)

def viterbi(states,transitions,emissions,sequence):
    F = initialize_matrix(len(states),len(sequence)+2)
    FP = initialize_matrix(len(states),len(sequence)+2,states[0])
    F[0][0] = 1
    for i in range(1,len(states)-1):
        F[i][1] = transitions[(states[0],states[i])]*emissions[states[i]][sequence[0]]
    for j in range(2,len(sequence)+1):
        for i in range(1,len(states)-1):
            values = []
            for k in range(1,len(states)-1):
                values.append(F[k][j-1]*transitions[(states[k],states[i])]*emissions[states[i]][sequence[j-1]])
            max_val, max_ind = get_max_val_ind(values)
            F[i][j] = max_val
            FP[i][j] = states[max_ind+1]
    values = []
    for k in range(1,len(states)-1):
        values.append(F[k][len(sequence)]*transitions[(states[k],states[-1])])
    max_val, max_ind = get_max_val_ind(values)
    F[-1][-1] = max_val
    FP[-1][-1] = states[max_ind+1]
    return F, FP

def traceback(states,FP):
    path = ['e'] # the last element of the path is the end state
    current = FP[-1][-1] # the current state is the one written in the last cell of the matrix
    for i in range(len(FP[0])-2,0,-1): # loops on the symbols
        path = [current] + path # appends the current state to the path
        current = FP[states.index(current)][i] # finds the index of the current state in the list of states and moves to the corresponing row of FP 
    path = ['b'] + path # the first element of the path is the begin state
    return ' '.join(path) # transforms the list into a string where elements are separated by spaces

path = traceback(states,FP)
print '- '+' '.join(sequence)+' -'
print path

