import warnings
from numpy import array, cos, dot, equal, kron, mod, pi, random, real,     reshape, sin, sqrt, zeros
from qutip import expect, basis, qeye, sigmax, sigmay, sigmaz, tensor
from scipy.optimize import minimize
from ncpol2sdpa import SdpRelaxation, generate_variables, flatten,     generate_measurements, projective_measurement_constraints

def correl_qubits(psi, sett, N, M, K, variables=None):
    '''Computes the correlations expected when some quantum state is
    measured according to some settings.
    '''
    # Setting up context and checking input
    nbInputs = len(sett)/2./N
    if nbInputs % 1 != 0:
        warnings.warn('Warning: Bad input for correl_qubits.', UserWarning)
    else:
        nbInputs = int(nbInputs)

    # Measurement operators definition
    c = [cos(v) for v in sett]
    s = [sin(v) for v in sett]

    A = [qeye(2)]
    B = [qeye(2)]
    C = [qeye(2)]
    for i in range(nbInputs):
        A.append((qeye(2) + s[2*i]*c[2*i+1]*sigmax() +
                  s[2*i]*s[2*i+1]*sigmay() + c[2*i]*sigmaz())/2)
        B.append((qeye(2) + s[2*i+2*nbInputs]*c[2*i+2*nbInputs+1]*sigmax() +
                  s[2*i+2*nbInputs]*s[2*i+2*nbInputs+1]*sigmay() +
                  c[2*i+2*nbInputs]*sigmaz())/2)
        C.append((qeye(2) + s[2*i+4*nbInputs]*c[2*i+4*nbInputs+1]*sigmax() +
                  s[2*i+4*nbInputs]*s[2*i+4*nbInputs+1]*sigmay() +
                  c[2*i+4*nbInputs]*sigmaz())/2)

    # Now we compute the multipartite operators.
    operators = [tensor(Ai, Bj, Ck) for Ai in A for Bj in B for Ck in C]
    probabilities = [expect(op, psi) for op in operators]
    if variables is not None:
        symb_vars = [Ai*Bj*Ck for Ai in flatten([1, variables[0]])
                     for Bj in flatten([1, variables[1]])
                     for Ck in flatten([1, variables[2]])]
        ret = {}
        for i, probability in enumerate(probabilities):
            ret[symb_vars[i]] = probability
    else:
        ret = probabilities

    return ret

def generate_substitutions(A, B, C):
    '''Defines additional substitution rules over the projectors to include
    biseparation and the independence of algebras.
    '''
    substitutions = {}
    # Biseparation
    for m1 in range(len(A[0])):
        for m2 in range(m1+1, len(A[0])):
            for k1 in range(len(A[0][m1])):
                for k2 in range(len(A[0][m2])):
                    substitutions[A[0][m2][k2]*A[0][m1][k1]] =                                   A[0][m1][k1]*A[0][m2][k2]
                    substitutions[B[1][m2][k2]*B[1][m1][k1]] =                                   B[1][m1][k1]*B[1][m2][k2]
                    substitutions[C[2][m2][k2]*C[2][m1][k1]] =                                   C[2][m1][k1]*C[2][m2][k2]

    # Independence of algebras
    for s1 in range(len(A)):
        for s2 in range(len(A)):
            if s1 != s2:
                for m1 in range(len(A[s1])):
                    for m2 in range(len(A[s2])):
                        for k1 in range(len(A[s1][m1])):
                            for k2 in range(len(A[s1][m1])):
                                substitutions[A[s1][m1][k1]*A[s2][m2][k2]] = 0
                                substitutions[B[s1][m1][k1]*B[s2][m2][k2]] = 0
                                substitutions[C[s1][m1][k1]*C[s2][m2][k2]] = 0
                                substitutions[A[s1][m1][k1]*B[s2][m2][k2]] = 0
                                substitutions[A[s1][m1][k1]*C[s2][m2][k2]] = 0
                                substitutions[B[s1][m1][k1]*C[s2][m2][k2]] = 0
                                substitutions[B[s1][m1][k1]*A[s2][m2][k2]] = 0
                                substitutions[C[s1][m1][k1]*A[s2][m2][k2]] = 0
                                substitutions[C[s1][m1][k1]*B[s2][m2][k2]] = 0
    return substitutions


def generate_equality_constraints(A, B, C, lamb, Prob):
    '''
    The correlation constraints are  equalities.
    '''
    S = len(A)
    M = len(A[0])
    K = len(A[0][0]) + 1
    eqs = []
    for m1 in range(M):
        for k1 in range(K-1):  # 1-partite marginals:
            eqs.append(sum(A[s][m1][k1] for s in range(S)) - ((1-lamb)*1/K +
                       lamb*Prob[A[0][m1][k1]]))
            eqs.append(sum(B[s][m1][k1] for s in range(S)) - ((1-lamb)*1/K +
                       lamb*Prob[B[0][m1][k1]]))
            eqs.append(sum(C[s][m1][k1] for s in range(S)) - ((1-lamb)*1/K +
                       lamb*Prob[C[0][m1][k1]]))
            for m2 in range(M):
                for k2 in range(K-1):  # 2-partite marginals:
                    eqs.append(sum(A[s][m1][k1]*B[s][m2][k2] for s in range(S))
                               - ((1-lamb)*1/(K**2) +
                               lamb*Prob[A[0][m1][k1]*B[0][m2][k2]]))
                    eqs.append(sum(A[s][m1][k1]*C[s][m2][k2] for s in range(S))
                               - ((1-lamb)*1/(K**2) +
                               lamb*Prob[A[0][m1][k1]*C[0][m2][k2]]))
                    eqs.append(sum(B[s][m1][k1]*C[s][m2][k2] for s in range(S))
                               - ((1-lamb)*1/(K**2) +
                               lamb*Prob[B[0][m1][k1]*C[0][m2][k2]]))
                    for m3 in range(M):
                        for k3 in range(K-1):  # joint probabilities:
                            eqs.append(
                              sum(A[s][m1][k1]*B[s][m2][k2]*C[s][m3][k3]
                              for s in range(S)) - ((1-lamb)*1/(K**3) +lamb*
                              Prob[A[0][m1][k1]*B[0][m2][k2]*C[0][m3][k3]]))
    return eqs

def get_relaxation(psi, settings, K, M, level, verbose=0):
    N = 3  # Number of parties
    partitions = ['BC|A', 'AC|B', 'AB|C']
    S = len(partitions)  # Number of possible partitions
    configuration = [K] * M
    # Noncommuting variables
    A = [[] for _ in range(S)]
    B = [[] for _ in range(S)]
    C = [[] for _ in range(S)]
    for s, partition in enumerate(partitions):
        A[s] = generate_measurements(configuration, 'A^%s_' % (partition))
        B[s] = generate_measurements(configuration, 'B^%s_' % (partition))
        C[s] = generate_measurements(configuration, 'C^%s_' % (partition))

    # Commuting, real-valued variable
    lambda_ = generate_variables('lambda')[0]

    # Obtain monomial substitutions to simplify the monomial basis
    substitutions = generate_substitutions(A, B, C)
    for s in range(S):
        substitutions.update(projective_measurement_constraints(A[s], B[s],
                                                                C[s]))

    if verbose > 0:
        print('Total number of substitutions: %s' % len(substitutions))
    probabilities = correl_qubits(psi, settings, N, M, K, [A[0], B[0], C[0]])
    # The probabilities enter the problem through equality constraints
    equalities = generate_equality_constraints(A, B, C, lambda_, probabilities)
    if verbose > 0:
        print('Total number of equality constraints: %s' % len(equalities))
    objective = -lambda_
    if verbose > 0:
        print('Objective function: %s' % objective)
    # Obtain SDP relaxation
    sdpRelaxation = SdpRelaxation(flatten([A, B, C]), parameters=[lambda_],
                                  verbose=verbose)
    sdpRelaxation.get_relaxation(level, objective=objective,
                                 momentequalities=equalities,
                                 substitutions=substitutions)
    variables = [A, B, C, lambda_]
    return variables, sdpRelaxation


def get_solution(variables, sdpRelaxation, psi, settings):
    M = len(variables[0][0])
    K = len(variables[0][0][0]) + 1
    probabilities = correl_qubits(psi, settings, 3, M, K, [variables[0][0],
                                                           variables[1][0],
                                                           variables[2][0]])
    equalities = generate_equality_constraints(variables[0], variables[1],
                                               variables[2], variables[3],
                                               probabilities)
    sdpRelaxation.process_constraints(momentequalities=equalities)
    sdpRelaxation.solve()
    return sdpRelaxation.primal

def funk(settings):
    value = get_solution(variables, sdpRelaxation, psi, settings)
    print(value)
    return value

N = 3      # Number of parties
M = 2      # Number of measuerment settings
K = 2      # Number of outcomes
level = 2  # Order of relaxation
psi = (tensor(basis(2, 0), basis(2, 0), basis(2, 0)) +
       tensor(basis(2, 1), basis(2, 1), basis(2, 1))).unit()  # GHZ state
settings = [pi/2, -pi/12, pi/2, -pi/12+pi/2, pi/2, -pi/12, pi/2,
            -pi/12+pi/2, pi/2, -pi/12, pi/2, -pi/12+pi/2]
result = minimize(funk, settings)

