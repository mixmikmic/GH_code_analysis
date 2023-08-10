import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import product
from math import sqrt, sin, cos, pi, atan
from qutip import tensor, basis, sigmax, sigmaz, expect, qeye
from ncpol2sdpa import SdpRelaxation, flatten, generate_measurements,                        projective_measurement_constraints
π = pi

def get_moments(ξ, θ):
    mu = atan(sin(2*θ))
    psi = (cos(θ) * tensor(basis(2, 0),basis(2, 0)) +
           sin(θ) * tensor(basis(2, 1),basis(2, 1)))
    A_1 = cos(mu)*sigmaz() - sin(mu)*sigmax()
    A_0 = cos(mu)*sigmaz() + sin(mu)*sigmax()

    B_0 = sigmaz()
    B_1 = (qeye(2) + cos(2*ξ)*sigmax())/2

    A_00 = (qeye(2) + A_0)/2
    A_10 = (qeye(2) + A_1)/2
    B_00 = (qeye(2) + B_0)/2
    B_10 = B_1

    p = []
    p.append(expect(tensor(A_00, qeye(2)), psi))
    p.append(expect(tensor(A_10, qeye(2)), psi))
    p.append(expect(tensor(qeye(2), B_00), psi))
    p.append(expect(tensor(qeye(2), B_10), psi))

    p.append(expect(tensor(A_00, B_00), psi))
    p.append(expect(tensor(A_00, B_10), psi))
    p.append(expect(tensor(A_10, B_00), psi))
    p.append(expect(tensor(A_10, B_10), psi))

    moments = ["-0[0,0]-1[0,0]+1"]
    k = 0
    for i in range(len(A_configuration)):
        moments.append(P_0_A[i][0] + P_1_A[i][0] - p[k])
        k += 1
    for i in range(len(B_configuration)):
        moments.append(P_0_B[i][0] + P_1_B[i][0] - p[k])
        k += 1
    for i in range(len(A_configuration)):
        for j in range(len(B_configuration)):
            moments.append(P_0_A[i][0]*P_0_B[j][0] + P_1_A[i][0]*P_1_B[j][0] - p[k])
            k += 1
    return moments

level = 4
A_configuration = [2, 2]
B_configuration = [2, 2]

P_0_A = generate_measurements(A_configuration, 'P_0_A')
P_0_B = generate_measurements(B_configuration, 'P_0_B')
P_1_A = generate_measurements(A_configuration, 'P_1_A')
P_1_B = generate_measurements(B_configuration, 'P_1_B')
substitutions = projective_measurement_constraints(P_0_A, P_0_B)
substitutions.update(projective_measurement_constraints(P_1_A, P_1_B))

guessing_probability = - (P_0_B[1][0] - P_1_B[1][0])
sdp = SdpRelaxation([flatten([P_0_A, P_0_B]), flatten([P_1_A, P_1_B])],
                    verbose=0, normalized=False)

def iterate_over_parameters(Ξ, Θ):
    result = []
    for ξ, θ in product(Ξ, Θ):
        if sdp.block_struct == []:
            sdp.get_relaxation(level, objective=guessing_probability,
                               momentequalities=get_moments(ξ, θ),
                               substitutions=substitutions,
                               extraobjexpr="-1[0,0]")
        else:
            sdp.process_constraints(momentequalities=get_moments(ξ, θ))
        sdp.solve(solver='sdpa', solverparameters={"executable": "sdpa_gmp",
                                                   "paramsfile": "param.gmp.sdpa"})
        result.append({"ξ": ξ, "θ": θ, "primal": sdp.primal, "dual": sdp.dual, "status": sdp.status})
    return result

def print_latex_table(results):
    range_ = set([result["θ"] for result in results])
    print("$\\xi$ & Bits \\\\")
    for θ in range_:
        print("$\\theta=%.3f$ &  \\\\" % θ)
        for result in results:
            if result["θ"] == θ:
                print("%.3f & %.3f\\\\" % 
                      (result["ξ"], abs(np.log2(-result["primal"]))))

                
def plot_results(results, labels, filename=None):
    domain = sorted(list(set(result["ξ"] for result in results)))
    range_ = sorted(list(set(result["θ"] for result in results)))
    fig, axes = plt.subplots(ncols=1)
    for i, θ in enumerate(range_):
        randomness = [abs(np.log2(-result["primal"])) 
                      for result in results if result["θ"] == θ]
        axes.plot(domain, randomness, label=labels[i])
    axes.set_xlabel("$ξ$")
    axes.set_ylabel("Randomness [bits]")
    axes.legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()

exponents = range(2, 6)
results = iterate_over_parameters(np.linspace(0, π/4, 60), 
                                  [π/2**i for i in exponents] + [0])

plot_results(results, ["$θ=0$"] + ["$θ=π/%d$" % 2**i for i in sorted(exponents, reverse=True)])

