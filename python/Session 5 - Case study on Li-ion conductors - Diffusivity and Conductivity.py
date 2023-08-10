import json
from monty.io import zopen

with zopen("trajectories.json.gz") as f:
    data = json.load(f)

print("There are %d documents." % len(data))
print("A sample document : ")
print(json.dumps(data[0], indent=4))

from pymongo import MongoClient, ASCENDING

# These few lines connects to the database that has been setup for this workshop.
client = MongoClient("ds141786.mlab.com", port=41786)
db = client.trajectories
db.authenticate("user", "miguest")
traj = db.trajectories

# These two lines insert the data into the MongoDB database.
# traj.ensure_index("step")
# traj.insert_many(data)

from pymatgen import Structure
from pymatgen.analysis.diffusion_analyzer import DiffusionAnalyzer
Na3PS4Cl = Structure.from_file("Cl_doped_Na3PS4/Na47P16S63Cl1_0/Na47P16S63Cl1.cif")
Na3PS4Cl.remove_oxidation_states()

def get_diffusivity_analyzer(temp, time_step, step_skip, use_db=True):
    
    species = Na3PS4Cl.species 
    if use_db:
        crit = {"step": {"$mod": [step_skip, 0], "$gte": 50000},
                "md_id": 116, "end_temp": temperature,
                "config_id": 1}
        structures = []
        for r in traj.find(crit, projection=["structure", "step"], sort=[("step", ASCENDING)]):
            structures.append(Structure(r["structure"]["lattice"], species, r["structure"]["frac_coords"]))
        a = DiffusionAnalyzer.from_structures(structures, "Na", temperature=temperature, 
                                              time_step=time_step, step_skip=step_skip, smoothed=False)
        return a
    else:
        # This is a backup that queries the data from the loaded file in case connection to the MongoDB
        # does not work during the workshop.
        temp_data = [d for d in data if d["end_temp"] == temp]
        temp_data.sort(key=lambda d: d["step"])
        structures = []
        for d in temp_data:
            structures.append(Structure(d["structure"]["lattice"], species, d["structure"]["frac_coords"]))
        a = DiffusionAnalyzer.from_structures(structures, "Na", temperature=temperature, 
                                              time_step=time_step, step_skip=step_skip, smoothed=False)
        return a

get_ipython().run_line_magic('matplotlib', 'inline')
step_skip = 20
time_step = 2
diffusivities = []
for temperature in [800, 1000, 1200]:
    da = get_diffusivity_analyzer(temperature, time_step, step_skip)
    da.plot_msd()
    diffusivities.append(da.diffusivity)

from pymatgen.analysis.diffusion_analyzer import get_arrhenius_plot, get_extrapolated_conductivity

plt = get_arrhenius_plot([800, 1000, 1200], diffusivities)

print("The extrapolated conductivity at 300K is %.1f mS/cm" % 
      get_extrapolated_conductivity([800, 1000, 1200], diffusivities, 300, Na3PS4Cl, "Na"))

