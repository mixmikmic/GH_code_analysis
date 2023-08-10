import os
import sys
import argparse

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from aiida.backends.utils import load_dbenv, is_dbenv_loaded
if not is_dbenv_loaded():
    load_dbenv()

from aiida.orm import CalculationFactory, DataFactory
from aiida.orm.code import Code
from aiida.orm.data.base import Int, Float, Str
from aiida.orm.data.upf import UpfData
from aiida.orm.data.structure import StructureData
from aiida.common.exceptions import NotExistent
from aiida.work.run import run, async, submit
from aiida.work.workchain import WorkChain, ToContext, while_, Outputs

from common.structure.generate import create_diamond_fcc, scale_structure
from common.pseudo.qe import get_pseudos

KpointsData = DataFactory("array.kpoints")
ParameterData = DataFactory('parameter')
PwCalculation = CalculationFactory('quantumespresso.pw')

get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

class EquationOfState(WorkChain):
    """
    Workchain that for a given structure will compute the equation of state by
    computing the total energy for a set of derived structures with a scaled
    lattice parameter
    """

    @classmethod
    def define(cls, spec):
        """
        This is the most important method of a Workchain, that defines the
        inputs that it takes, the logic of the execution and the outputs
        that are generated in the process 
        """
        super(EquationOfState, cls).define(spec)
        
        # First we define the inputs, specifying the type we expect
        spec.input("structure", valid_type=StructureData)
        spec.input("codename", valid_type=Str)
        spec.input("pseudo_family", valid_type=Str)
        spec.input("npoints", valid_type=Int)
        
        # The outline describes the business logic that defines
        # which steps are executed in what order and based on
        # what conditions. Each `cls.method` is implemented below
        spec.outline(
            cls.init,
            while_(cls.should_run_pw)(
                cls.run_pw,
                cls.parse_pw,
            ),
            cls.return_result,
        )
        
        # Here we define the output the Workchain will generate and
        # return. Dynamic output allows a variety of AiiDA data nodes
        # to be returned
        spec.dynamic_output()

    def init(self):
        """
        Initialize variables and the scales we want to compute
        """
        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([2, 2, 2])

        npoints = self.inputs.npoints.value
        self.ctx.i = 0
        self.ctx.scales = sorted([1 - pow(-1, x)*0.02*int((x+1)/2) for x in range(npoints)])
        self.ctx.result = []
        self.ctx.options = {
            "resources": {
                "num_machines": 1,
                "tot_num_mpiprocs": 1,
            },
            "max_wallclock_seconds": 30 * 60,
        }
        self.ctx.parameters = {
            "CONTROL": {
                "calculation": "scf",
                "tstress": True,
                "tprnfor": True,
            },
            "SYSTEM": {
                "ecutwfc": 30.,
                "ecutrho": 200.,
            },
            "ELECTRONS": {
                "conv_thr": 1.e-6,
            }
        }
        self.ctx.kpoints = kpoints
        self.ctx.pseudos = get_pseudos(self.inputs.structure, self.inputs.pseudo_family.value)
        
        # Initialize plot variables
        self.fig, self.ax = plt.subplots(1,1)
        self.ax.set_xlabel(u"Volume [Å^3]")
        self.ax.set_ylabel(u"Total energy [eV]")

    def should_run_pw(self):
        """
        This is the main condition of the while loop, as defined
        in the outline of the Workchain. We only run another
        pw.x calculation if the current iteration is smaller than
        the total number of scale factors we want to compute
        """
        return self.ctx.i < len(self.ctx.scales)

    def run_pw(self):
        """
        This is the main function that will perform a pw.x
        calculation for the current scaling factor
        """
        scale = self.ctx.scales[self.ctx.i]
        structure = scale_structure(self.inputs.structure, Float(scale))
        self.ctx.i += 1

        # Create the input dictionary
        inputs = {
            'code'       : Code.get_from_string(self.inputs.codename.value),
            'pseudo'     : self.ctx.pseudos,
            'kpoints'    : self.ctx.kpoints,
            'structure'  : structure,
            'parameters' : ParameterData(dict=self.ctx.parameters),
            '_options'   : self.ctx.options,
        }

        # Create the calculation process and launch it
        self.report("Running pw.x for the scale factor {}".format(scale))
        process = PwCalculation.process()
        future  = async(process, **inputs)

        return ToContext(pw=Outputs(future))

    def parse_pw(self):
        """
        Extract the volume and total energy of the last completed PwCalculation
        """
        volume = self.ctx.pw["output_parameters"].dict.volume
        energy = self.ctx.pw["output_parameters"].dict.energy
        self.ctx.result.append((volume, energy))
        
        self.plot_data()

    def return_result(self):
        """
        Attach the results of the PwCalculations and the initial structure to the outputs
        """
        result = {
            "initial_structure": self.inputs.structure,
            "result": ParameterData(dict={"eos": self.ctx.result}),
        }

        for link_name, node in result.iteritems():
            self.out(link_name, node)

        self.report("Workchain <{}> completed successfully".format(self.calc.pk))

    def plot_data(self):
        self.ax.plot(*zip(*self.ctx.result), marker='o', linestyle='--', color='r')
        self.fig.canvas.draw()

# Input variables
element='C'
alat=3.65
npoints=5

# Create the starting structure
structure = create_diamond_fcc(Str(element), Float(alat))

# Define the code and pseudo family to be used
codename='pw-5.1'
pseudo_family='SSSP'

outputs = run(
    EquationOfState,
    npoints=Int(npoints),
    structure=structure,
    codename=Str(codename),
    pseudo_family=Str(pseudo_family)
)

print "\nFinal results of the equation of state workchain:\n"
print "{volume:12}  {energy:12}".format(volume="Volume (A^3)", energy="Energy (eV)")
print "{}".format("-"*26)
for p in outputs["result"].get_dict()['eos']:
    print "{volume:>12.5f}  {energy:>12.5f}".format(volume=p[0], energy=p[1])



