from ipywidgets import (RadioButtons, VBox, HBox, Layout, Label, Checkbox, Text, IntSlider)
from qgrid import show_grid
label_layout = Layout(width='100%')
from isatools.create.models import *
from isatools.model import Investigation
from isatools.isatab import dump_tables_to_dataframes as dumpdf
import qgrid
qgrid.nbinstall(overwrite=True)

rad_study_design = RadioButtons(options=['Intervention', 'Observation'], value='Intervention', disabled=False)
VBox([Label('Study design type?', layout=label_layout), rad_study_design])

if rad_study_design.value == 'Intervention':
    study_design = InterventionStudyDesign()
if rad_study_design.value == 'Observation':
    study_design = None
intervention_ui_disabled = not isinstance(study_design, InterventionStudyDesign)
intervention_type = RadioButtons(options=['single', 'multiple'], value='single', disabled=intervention_ui_disabled)
intervention_type_vbox = VBox([Label('Single intervention or to multiple intervention?', layout=label_layout), intervention_type])
free_or_restricted_design = RadioButtons(options=['yes', 'no'], value='no', disabled=intervention_ui_disabled)
free_or_restricted_design_vbox = VBox([Label("Are there 'hard to change' factors?", layout=label_layout), free_or_restricted_design])
HBox([intervention_type_vbox, free_or_restricted_design_vbox])

hard_to_change_factors_ui_disabled = free_or_restricted_design.value == 'no'
hard_to_change_factors = RadioButtons(options=[1, 2], value=1, disabled=hard_to_change_factors_ui_disabled)
VBox([Label("If applicable, how many 'hard to change factors'?", layout=label_layout), hard_to_change_factors])

repeats = intervention_type.value != 'single'
factorial_design = free_or_restricted_design.value == 'no'
split_plot_design = (free_or_restricted_design.value == 'yes' and hard_to_change_factors.value == 1)
split_split_plot_design = (free_or_restricted_design.value == 'yes' and hard_to_change_factors.value == 2)
print('Interventions: {}'.format('Multiple interventions' if repeats else 'Single intervention'))
design_type = 'factorial'  # always default to factorial
if split_plot_design:
    design_type = 'split plot'
elif split_split_plot_design:
    design_type = 'split split'
print('Design type: {}'.format(design_type))

factorial_design_ui_disabled = not factorial_design
chemical_intervention = Checkbox(value=True, description='Chemical intervention', disabled=factorial_design_ui_disabled)
behavioural_intervention = Checkbox(value=False, description='Behavioural intervention', disabled=factorial_design_ui_disabled)
surgical_intervention = Checkbox(value=False, description='Surgical intervention', disabled=factorial_design_ui_disabled)
biological_intervention = Checkbox(value=False, description='Biological intervention', disabled=factorial_design_ui_disabled)
radiological_intervention = Checkbox(value=False, description='Radiological intervention', disabled=factorial_design_ui_disabled)
VBox([chemical_intervention, surgical_intervention, biological_intervention, radiological_intervention])

level_uis = []
if chemical_intervention:
    agent_levels = Text(
        value='calpol, no agent',
        placeholder='e.g. cocaine, calpol',
        description='Agent:',
        disabled=False
    )
    dose_levels = Text(
        value='low, high',
        placeholder='e.g. low, high',
        description='Dose levels:',
        disabled=False
    )
    duration_of_exposure_levels = Text(
        value='short, long',
        placeholder='e.g. short, long',
        description='Duration of exposure:',
        disabled=False
    )
VBox([Label("Chemical intervention factor levels:", layout=label_layout), agent_levels, dose_levels, duration_of_exposure_levels])

factory = TreatmentFactory(intervention_type=INTERVENTIONS['CHEMICAL'], factors=BASE_FACTORS)
for agent_level in agent_levels.value.split(','):
    factory.add_factor_value(BASE_FACTORS[0], agent_level.strip())
for dose_level in dose_levels.value.split(','):
    factory.add_factor_value(BASE_FACTORS[1], dose_level.strip())
for duration_of_exposure_level in duration_of_exposure_levels.value.split(','):
    factory.add_factor_value(BASE_FACTORS[2], duration_of_exposure_level.strip())
print('Number of study groups (treatment groups): {}'.format(len(factory.compute_full_factorial_design())))
treatment_sequence = TreatmentSequence(ranked_treatments=factory.compute_full_factorial_design())

group_blanced = RadioButtons(options=['Balanced', 'Unbalanced'], value='Balanced', disabled=False)
VBox([Label('Are study groups balanced?', layout=label_layout), group_blanced])

group_size = IntSlider(value=5, min=0, max=100, step=1, description='Group size:', disabled=False, continuous_update=False, orientation='horizontal', readout=True, readout_format='d')
group_size

plan = SampleAssayPlan(group_size=group_size.value)

rad_sample_type = RadioButtons(options=['Blood', 'Sweat', 'Tears', 'Urine'], value='Blood', disabled=False)
VBox([Label('Sample type?', layout=label_layout), rad_sample_type])

sampling_size = IntSlider(value=3, min=0, max=100, step=1, description='Sample size:', disabled=False, continuous_update=False, orientation='horizontal', readout=True, readout_format='d')
sampling_size

inject_qc_material = Text(
        value='solvent blank',
        placeholder='e.g. blank, water, solvent',
        description='Material:',
        disabled=False
    )
inject_qc_interval = IntSlider(value=10, min=0, max=10, step=1, description='Injection interval:', disabled=False, continuous_update=False, orientation='horizontal', readout=True, readout_format='d')
VBox([inject_qc_material, inject_qc_interval])

plan.add_sample_type(rad_sample_type.value)
plan.add_sample_plan_record(rad_sample_type.value, sampling_size.value)
plan.add_sample_qc_plan_record(inject_qc_material.value, inject_qc_interval.value)
plan.add_sample_qc_plan_record('reference material', 5)
isa_object_factory = IsaModelObjectFactory(plan, treatment_sequence)

import json
from isatools.create.models import SampleAssayPlanEncoder
print(json.dumps(plan, cls=SampleAssayPlanEncoder, sort_keys=True, indent=4, separators=(',', ': ')))

isa_investigation = Investigation(identifier='inv101')
isa_study = isa_object_factory.create_study_from_plan()
isa_study.filename = 's_study.txt'
isa_investigation.studies = [isa_study]
dataframes = dumpdf(isa_investigation)
sample_table = next(iter(dataframes.values()))
sample_table

print('Total rows generated: {}'.format(len(sample_table)))



