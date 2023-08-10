import pycosat

def gen_all_clauses(k, n):
    if k == 0:
        yield tuple()
    else:
        for clause in gen_all_clauses(k-1, n):
            for var in range(1, n+1):
                for svar in [var, -var]:
                    yield clause + (svar,)

def gen_all_kcnf(k, n, m):
    if m == 0:
        yield tuple()
    else:
        for cnf in gen_all_kcnf(k, n, m-1):
            for clause in gen_all_clauses(k, n):
                yield cnf + (clause,)

def filter_satable(cnf_gen):
    for cnf in cnf_gen:
        if pycosat.solve(cnf) != 'UNSAT':
            yield cnf

def filter_satable_by(cnf_gen, solution):
    # WARNING: it assumes solution is full
    solution_clauses = tuple([svar] for svar in solution)
    for cnf in cnf_gen:
        if pycosat.solve(cnf + solution_clauses) != 'UNSAT':
            yield cnf

def gen_satable_by_clause(k, n, solution):
    if k == 0:
        return
    else:
        for svar in solution:
            for clause in gen_all_clauses(k-1, n):
                yield clause + (svar,)
        for clause in gen_satable_by_clause(k-1, n, solution):
            for svar in solution:
                yield clause + (-svar,)

def gen_satable_by_kcnf(k, n, m, solution):
    if m == 0:
        yield tuple()
    else:
        for cnf in gen_satable_by_kcnf(k, n, m-1, solution):
            for clause in gen_satable_by_clause(k, n, solution):
                yield cnf + (clause,)

K = 2
VARS = 1
CLAUSES = 1
SOLUTION = [x for x in range(1, VARS+1)]

all_possible = tuple(sorted(list(gen_all_kcnf(K, VARS, CLAUSES))))
all_satable = tuple(filter_satable(all_possible))
all_satable_by = tuple(filter_satable_by(all_possible, SOLUTION))
print("Possible:", len(all_possible))
print("Satable:", len(all_satable))
print("Satable by:", len(all_satable_by))
print((2**K - 1) * VARS**K)
print((2**K) * VARS**K)

all_satable_by = tuple(sorted(tuple(gen_satable_by_kcnf(K, VARS, CLAUSES, SOLUTION))))
print("Satable by:", len(all_satable_by))

from cnf import get_random_sat_kcnfs

cnfs, solutions = get_random_sat_kcnfs(10000, 3, 10, 200)
sat_cnfs = [cnf for cnf in cnfs if cnf.satisfiable()]
print(len(cnfs), len(sat_cnfs))
assert cnfs == sat_cnfs
print(len(cnfs))

