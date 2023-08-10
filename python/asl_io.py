# %load script.py
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition

def create_model():
    model = ConcreteModel()
    model.x = Var()
    model.o = Objective(expr=model.x)
    model.c = Constraint(expr=model.x >= 1)
    model.x.set_value(1.0)
    return model

if __name__ == "__main__":

    with SolverFactory("ipopt") as opt:
        model = create_model()
        results = opt.solve(model, load_solutions=False)
        if results.solver.termination_condition != TerminationCondition.optimal:
            raise RuntimeError('Solver did not report optimality:\n%s'
                               % (results.solver))
        model.solutions.load_from(results)
        print("Objective: %s" % (model.o()))

# %load write.py
import pyomo.environ
from pyomo.core import ComponentUID
from pyomo.opt import ProblemFormat
# use fast version of pickle (python 2 or 3)
from six.moves import cPickle as pickle

def write_nl(model, nl_filename, **kwds):
    """
    Writes a Pyomo model in NL file format and stores
    information about the symbol map that allows it to be
    recovered at a later time for a Pyomo model with
    matching component names.
    """
    symbol_map_filename = nl_filename+".symbol_map.pickle"

    # write the model and obtain the symbol_map
    _, smap_id = model.write(nl_filename,
                             format=ProblemFormat.nl,
                             io_options=kwds)
    symbol_map = model.solutions.symbol_map[smap_id]

    # save a persistent form of the symbol_map (using pickle) by
    # storing the NL file label with a ComponentUID, which is
    # an efficient lookup code for model components (created
    # by John Siirola)
    tmp_buffer = {} # this makes the process faster
    symbol_cuid_pairs = tuple(
        (symbol, ComponentUID(var_weakref(), cuid_buffer=tmp_buffer))
        for symbol, var_weakref in symbol_map.bySymbol.items())
    with open(symbol_map_filename, "wb") as f:
        pickle.dump(symbol_cuid_pairs, f)

    return symbol_map_filename

if __name__ == "__main__":
    from script import create_model

    model = create_model()
    nl_filename = "example.nl"
    symbol_map_filename = write_nl(model, nl_filename)
    print("        NL File: %s" % (nl_filename))
    print("Symbol Map File: %s" % (symbol_map_filename))

get_ipython().run_cell_magic('bash', '', 'ipopt -s example.nl')

# %load read.py
import pyomo.environ
from pyomo.core import SymbolMap
from pyomo.opt import (ReaderFactory,
                       ResultsFormat)
# use fast version of pickle (python 2 or 3)
from six.moves import cPickle as pickle

def read_sol(model, sol_filename, symbol_map_filename, suffixes=[".*"]):
    """
    Reads the solution from the SOL file and generates a
    results object with an appropriate symbol map for
    loading it into the given Pyomo model. By default all
    suffixes found in the NL file will be extracted. This
    can be overridden using the suffixes keyword, which
    should be a list of suffix names or regular expressions
    (or None).
    """
    if suffixes is None:
        suffixes = []

    # parse the SOL file
    with ReaderFactory(ResultsFormat.sol) as reader:
        results = reader(sol_filename, suffixes=suffixes)

    # regenerate the symbol_map for this model
    with open(symbol_map_filename, "rb") as f:
        symbol_cuid_pairs = pickle.load(f)
    symbol_map = SymbolMap()
    symbol_map.addSymbols((cuid.find_component(model), symbol)
                          for symbol, cuid in symbol_cuid_pairs)

    # tag the results object with the symbol_map
    results._smap = symbol_map

    return results

if __name__ == "__main__":
    from pyomo.opt import TerminationCondition
    from script import create_model

    model = create_model()
    sol_filename = "example.sol"
    symbol_map_filename = "example.nl.symbol_map.pickle"
    results = read_sol(model, sol_filename, symbol_map_filename)
    if results.solver.termination_condition !=        TerminationCondition.optimal:
        raise RuntimeError("Solver did not terminate with status = optimal")
    model.solutions.load_from(results)
    print("Objective: %s" % (model.o()))

