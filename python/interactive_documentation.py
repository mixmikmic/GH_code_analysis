import pyrosetta.distributed.docs as docs
import pyrosetta

get_ipython().run_line_magic('psearch', 'pyrosetta.rosetta.protocols.simple_moves.* all')

get_ipython().run_line_magic('pinfo', 'pyrosetta.rosetta.protocols.simple_moves.ChangeAndResetFoldTreeMover')

print(len(docs.movers._component_names))

_ = list(map(print, docs.movers._component_names))

docs.mover.CoMTrackerCM

get_ipython().run_line_magic('psearch', 'docs.movers.*KIC')

docs.movers.GeneralizedKIC

docs.mover.SecretSauceMover

