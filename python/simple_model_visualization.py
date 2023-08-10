from cobra.core import Metabolite, Reaction, Model

model = Model('simple_model')

A = Metabolite('A')
B = Metabolite('B')
C = Metabolite('C')
D = Metabolite('D')
E = Metabolite('E')
P = Metabolite('P')

R1 = Reaction('R1')
R2 = Reaction('R2')
R3 = Reaction('R3')
R4 = Reaction('R4')
R5 = Reaction('R5')
R6 = Reaction('R6')
R7 = Reaction('R7')
R8 = Reaction('R8')
R9 = Reaction('R9')
R10 = Reaction('R10')

model.add_metabolites([A, B, C, D, E, P])
model.add_reactions([R1, R2, R3, R4, R5, R6, R7, R8, R9, R10])

model.reactions.R1.build_reaction_from_string('--> A')
model.reactions.R2.build_reaction_from_string('<--> B')
model.reactions.R3.build_reaction_from_string('P -->')
model.reactions.R4.build_reaction_from_string('E -->')
model.reactions.R5.build_reaction_from_string('A --> B')
model.reactions.R6.build_reaction_from_string('A --> C')
model.reactions.R7.build_reaction_from_string('A --> D')
model.reactions.R8.build_reaction_from_string('B <--> C')
model.reactions.R9.build_reaction_from_string('B --> P')
model.reactions.R10.build_reaction_from_string('C + D --> E + P')

from d3flux import flux_map
flux_map(model, display_name_format=lambda x: str(x.id), figsize=(300,250))

A.notes['map_info']['x'] = 150.
A.notes['map_info']['y'] = 50.

B.notes['map_info']['x'] = 80.
B.notes['map_info']['y'] = 130.

C.notes['map_info']['x'] = 150.
C.notes['map_info']['y'] = 130.

D.notes['map_info']['x'] = 220.
D.notes['map_info']['y'] = 130.

P.notes['map_info']['x'] = 150.
P.notes['map_info']['y'] = 200.

E.notes['map_info']['x'] = 220.
E.notes['map_info']['y'] = 200.

flux_map(model, display_name_format=lambda x: str(x.id), figsize=(300,250))

from cobra.io import load_json_model
new_model = load_json_model('simple_model.json')
flux_map(new_model, figsize=(300,250))

new_model.objective = model.reactions.R4
new_model.optimize()

flux_map(new_model, figsize=(300,250))

new_model.reactions.R8.knock_out()
new_model.optimize()
flux_map(new_model, figsize=(300,250))



