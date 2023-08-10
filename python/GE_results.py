import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

def count_nodes(s): return 1 + s.count(",") + s.count("(")

filenames = ["GE_results.dat"]
problems = [
    "Pagie2D",
    "DowNorm",
    "HousingNorm",
    "TowerNorm",
    "Vladislavleva4"
    ]
generators = ["BNF", "exec"]

for filename in filenames:
    d = pd.read_csv(filename, delimiter="\t", 
                    names=["problem", "grammar", "solver", "generator", "str_trace", "budget", "rep", "obj", "test_obj", "fn"])
    # shorten the names
    d.loc[d["generator"] == "GE_randsol", "generator"] = "BNF"
    d.loc[d["generator"] == "GE_randsol_sr_nobnf", "generator"] = "exec"
    d["node_count"] = d.apply(lambda row: count_nodes(row.fn), axis=1)
    
    for problem in problems:
        for generator in generators:
            d[(d["problem"] == problem) & (d["generator"] == generator)].boxplot(
                column="obj", by=["solver", "str_trace"], grid=False)
            plt.title(": ".join(("GE", problem, generator)))
            plt.suptitle("")
            plt.ylabel("Objective")
            
            d[(d["problem"] == problem) & (d["generator"] == generator)].boxplot(
                column="node_count", by=["solver", "str_trace"], grid=False)
            plt.title(": ".join(("GE", problem, generator)))
            plt.suptitle("")
            plt.ylabel("Node count")
            

# in the experiment with duplicate avoidance,
# we only used the "executable" grammar, not the bnf
filenames = ["GE_duplicate_results.dat"]
generators = ["exec"]

for filename in filenames:
    d = pd.read_csv(filename, delimiter="\t", 
                    names=["problem", "grammar", "solver", "generator", "str_trace", "budget", "rep", "obj", "test_obj", "fn"])
    # shorten the names
    d.loc[d["generator"] == "GE_randsol", "generator"] = "BNF"
    d.loc[d["generator"] == "GE_randsol_sr_nobnf", "generator"] = "exec"
    d["node_count"] = d.apply(lambda row: count_nodes(row.fn), axis=1)
    
    for problem in problems:
        for generator in generators:
            d[(d["problem"] == problem) & (d["generator"] == generator)].boxplot(
                column="obj", by=["solver", "str_trace"], grid=False)
            plt.title(": ".join(("GE no dups", problem, generator)))
            plt.suptitle("")
            plt.ylabel("Objective")
            
            d[(d["problem"] == problem) & (d["generator"] == generator)].boxplot(
                column="node_count", by=["solver", "str_trace"], grid=False)
            plt.title(": ".join(("GE no dups", problem, generator)))
            plt.suptitle("")
            plt.ylabel("Node count")
            

dirname = "/Users/jmmcd/Desktop/GE_duplicate_results_gens"

problems = [
    "Pagie2D",
    "DowNorm",
    "HousingNorm",
    "TowerNorm",
    "Vladislavleva4"
    ]
grammar_file = "sr.bnf"
generators = ["GE_randsol_sr_nobnf"]
solvers = ["RS", "HC", "LA", "EA"]
str_traces = [False, True]
reps = 30
budget = 20000

for problem in problems:
    for generator in generators:
        for str_trace in str_traces:
            for solver in solvers:
                basename = "_".join((problem, grammar_file, solver, generator, str(int(str_trace)), str(budget)))
                filenames = [basename + "_" + str(rep) + ".gens" for rep in range(reps)]
                
                d = np.array([np.genfromtxt(os.path.join(dirname, filename))[:budget, :] for filename in filenames])
                d = d.mean(axis=0)
                plt.plot(d[:, 0], d[:, 1], label=solver)
                
            plt.title(": ".join((problem, "Structured" if str_trace else "Linear", generator)))
            plt.xlabel("Iteration")
            plt.ylabel("Objective")
            plt.legend()
            plt.show()

filename = "/Users/jmmcd/Desktop/GE_results_poly.dat"
generators = ["exec"]
problems = ["poly_%d_%d" % (d, n) for d in range(2, 21, 2) for n in range(1, 4)]
solvers = ["RS", "HC", "EA"]

d = pd.read_csv(filename, delimiter="\t", 
                names=["problem", "grammar", "solver", "generator", "str_trace", "budget", "rep", "obj", "test_obj", "fn"])
# shorten the names
d.loc[d["generator"] == "GE_randsol", "generator"] = "BNF"
d.loc[d["generator"] == "GE_randsol_sr_nobnf", "generator"] = "exec"
d["node_count"] = d.apply(lambda row: count_nodes(row.fn), axis=1)

d.head()

for n in range(1, 4):
    for str_trace in [False, True]:
        for field in ["obj", "node_count"]:
            for solver in solvers:
                x = list(range(2, 21, 2))
                y = [d[
                    (d["problem"] == "poly_%d_%d" % (deg, n)) & 
                    (d["solver"] == solver) &
                    (d["str_trace"] == str_trace)
                      ][field].mean() for deg in x]
                plt.plot(x, y, label=solver)
                plt.title(": ".join(("GE", "poly", str(n), "vars", "str trace", str(str_trace))))
                if field == "obj": plt.ylabel("Objective")
                else: plt.ylabel("Node count")
                plt.xlabel("Degree")
                plt.xticks(range(2, 21, 2))
                plt.legend()
            plt.show()

filename = "GE_results_poly_scale_nvars.dat"
generators = ["exec"]
solvers = ["RS", "HC", "EA"]

d = pd.read_csv(filename, delimiter="\t", 
                names=["problem", "grammar", "solver", "generator", "str_trace", "budget", "rep", "obj", "test_obj", "fn"])
# shorten the names
d.loc[d["generator"] == "GE_randsol", "generator"] = "BNF"
d.loc[d["generator"] == "GE_randsol_sr_nobnf", "generator"] = "exec"
d["node_count"] = d.apply(lambda row: count_nodes(row.fn), axis=1)

d.head()

col = {"RS": "#FF6666", "HC": "#44BB44", "EA": "#222288"}
sty = {False: ":", True: "-"}
wid = {"RS": 2, "HC": 3, "EA": 4}

setups = (("RS", False), ("RS", True), ("HC", False), ("HC", True), ("EA", False), ("EA", True))
deg = 4
for solver, str_trace in setups:
    field = "obj"
    x = list(range(1, 11))
    y = [d[
           (d["problem"] == "poly_%d_%d" % (deg, n)) & 
           (d["solver"] == solver) &
           (d["str_trace"] == str_trace)
          ][field].mean() for n in x]
    yerr = [d[
           (d["problem"] == "poly_%d_%d" % (deg, n)) & 
           (d["solver"] == solver) &
           (d["str_trace"] == str_trace)
          ][field].std() for n in x]
    #sns.pointplot(x, y, ci=yerr, dodge=True, label="%s, %s" % (solver, ("Struct" if str_trace else "Lin")))
    plt.errorbar(x, y, yerr=None, lw=wid[solver], linestyle=sty[str_trace], c=col[solver], alpha=.7, label="%s/%s" % (solver, ("S" if str_trace else "L")))
    plt.title("")
    if field == "obj": plt.ylabel("Objective")
    else: plt.ylabel("Node count")
plt.xlabel(r"Number of variables $n$")
plt.xticks(range(1, 11))
plt.legend()
fig = plt.gcf()
fig.set_size_inches(4.2, 2.25)
plt.tight_layout()
plt.savefig("img/GE_polynomials_deg_4_scale_nvars.pdf")
plt.show()

filename = "GE_results.dat"
generator = "exec"
problems = [
    "Pagie2D",
    "Vladislavleva4",
    "DowNorm",
    "TowerNorm",
    "HousingNorm",
    ]
prob_names = {
    "Pagie2D": "P-2",
    "DowNorm": "Dow",
    "TowerNorm": "Tow",
    "Vladislavleva4": "V-4",
    "HousingNorm": "Hous",   
}
d = pd.read_csv(filename, delimiter="\t", 
                    names=["problem", "grammar", "solver", "generator", "str_trace", "budget", "rep", "obj", "test_obj", "fn"])
# shorten the names
d.loc[d["generator"] == "GE_randsol", "generator"] = "BNF"
d.loc[d["generator"] == "GE_randsol_sr_nobnf", "generator"] = "exec"
d["node_count"] = d.apply(lambda row: count_nodes(row.fn), axis=1)

field = "obj"
for solver, str_trace in setups: 
    print(solver, str_trace)
    x = list(range(len(problems)))
    y = [d[
           (d["generator"] == generator) & 
           (d["problem"] == problem) & 
           (d["solver"] == solver) &
           (d["str_trace"] == str_trace)
          ][field].mean()
        for problem in problems]

    print(y)
   
    #y = y.mean()
    plt.errorbar(x, y, yerr=None, lw=wid[solver], linestyle=sty[str_trace], c=col[solver], alpha=.7, label="%s/%s" % (solver, ("Structured" if str_trace else "Linear")))
plt.xlabel("Problem")
plt.xticks(x, [prob_names[problem] for problem in problems])
#plt.legend()
fig = plt.gcf()
fig.set_size_inches(2.25, 2.25)
plt.tight_layout()
plt.savefig("img/GE_dataset_problems.pdf")
plt.show()    

deg = 4

for str_trace in [False, True]:
    for field in ["obj", "node_count"]:
        for solver in solvers:
            x = list(range(1, 11))
            y = [d[
                (d["problem"] == "poly_%d_%d" % (deg, n)) & 
                (d["solver"] == solver) &
                (d["str_trace"] == str_trace)
                  ][field].mean() for n in x]
            plt.plot(x, y, label=solver)
            plt.title(": ".join(("GE", "poly", str(deg), "deg", "str trace", str(str_trace))))
            if field == "obj": plt.ylabel("Objective")
            else: plt.ylabel("Node count")
            plt.xlabel("Degree")
            plt.xticks(range(1, 11))
            plt.legend()
        plt.show()

dirname = "/Users/jmmcd/Desktop/GE_results_poly_scale_nvars"

ns = range(1, 11)
deg = 4
grammar_file = "sr.bnf"
generators = ["GE_randsol_sr_nobnf"]
solvers = ["RS", "HC", "LA", "EA"]
str_traces = [False, True]
reps = 30
budget = 20000

for n in ns:
    for generator in generators:
        for str_trace in str_traces:
            for solver in solvers:
                problem = "poly_%d_%d" % (deg, n)

                basename = "_".join((problem, grammar_file, solver, generator, str(int(str_trace)), str(budget)))
                filenames = [basename + "_" + str(rep) + ".gens" for rep in range(reps)]
                
                d = np.array([np.genfromtxt(os.path.join(dirname, filename))[:budget, :] for filename in filenames])
                d = d.mean(axis=0)
                plt.plot(d[:, 0], d[:, 1], label=solver)
                
            plt.title(": ".join((problem, "Structured" if str_trace else "Linear", generator)))
            plt.xlabel("Iteration")
            plt.ylabel("Objective")
            plt.legend()
            plt.show()



