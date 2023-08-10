# First, import required and useful Python modules.
# You can ignore this cell.

import copy
import datetime
import json

from IPython.display import display
from IPython.html import widgets
from IPython.lib.pretty import pprint
from IPython.utils.traitlets import Unicode
from openfisca_core import columns, conv, formulas, legislationsxml, reforms
import openfisca_france
from openfisca_france import entities, model

TaxBenefitSystem = openfisca_france.init_country()
tax_benefit_system = TaxBenefitSystem()

# Just run this cell with Ctrl-Enter

class FileWidget(widgets.DOMWidget):
    _view_name = Unicode('FilePickerView', sync=True)
    value = Unicode(sync=True)
    filename = Unicode(sync=True)
    
    def __init__(self, **kwargs):
        widgets.DOMWidget.__init__(self, **kwargs) # Call the base.
        self.errors = widgets.CallbackDispatcher(accepted_nargs=[0, 1])
        self.on_msg(self._handle_custom_msg)

    def _handle_custom_msg(self, content):
        if 'event' in content and content['event'] == 'error':
            self.errors()
            self.errors(self)

get_ipython().run_cell_magic('javascript', '', '\n// Just run this cell with Ctrl-Enter\n\nrequire(["widgets/js/widget"], function(WidgetManager){\n    var FilePickerView = IPython.WidgetView.extend({\n        render: function() {\n            this.setElement($(\'<input />\').attr(\'type\', \'file\'));\n        },\n        events: {\'change\': \'handle_file_change\'},\n        handle_file_change: function(evt) { \n            var file = evt.target.files[0];\n            if (file) {\n                var that = this;\n                var file_reader = new FileReader();\n                file_reader.onload = function(e) {\n                    that.model.set(\'value\', e.target.result);\n                    that.touch();\n                }\n                file_reader.readAsText(file);\n            } else {\n                this.send({ \'event\': \'error\' });\n            }\n            this.model.set(\'filename\', file.name);\n            this.touch();\n        },\n    });\n    WidgetManager.register_widget_view(\'FilePickerView\', FilePickerView);\n});')

# Just run this cell with Ctrl-Enter

reform_legislation_json = None
file_widget = FileWidget()
def file_loading():
    print("Loading %s" % file_widget.filename)
file_widget.on_trait_change(file_loading, 'filename')
def file_loaded():
    global reform_legislation_json
    reform_legislation_xml = file_widget.value
    reform_legislation_json, error = legislationsxml.xml_legislation_str_to_json(reform_legislation_xml)
    print(
        u'XML file loaded successfully' if error is None else u'XML file loading has failed: {}'.format(error)
        )
file_widget.on_trait_change(file_loaded, 'value')
def file_failed(*args):
    print("Could not load file contents of %s" % file_widget.filename)
file_widget.errors.register_callback(file_failed)
file_widget

def build_reform_1(tax_benefit_system):
    return reforms.Reform(
        entity_class_by_key_plural = entities.entity_class_by_key_plural,  # Keep the reference entities.
        legislation_json = reform_legislation_json,  # Was generated from the XML file you uploaded.
        name = u'Dummy reform 1 (legislation only)',
        reference = tax_benefit_system,
        )

dummy_reform_1 = build_reform_1(tax_benefit_system)  # tax_benefit_system was initialized above.

scenario_1 = dummy_reform_1.new_scenario().init_single_entity(
    period = 2014,
    parent1 = dict(
        birth = datetime.date(1974, 1, 1),
        sali = 5000,
        ),
    )

reference_simulation_1 = scenario_1.new_simulation(reference=True)
reform_simulation_1 = scenario_1.new_simulation()

# Calculate the "Revenu disponible" for both reference and reform simulations.
reference_revdisp_1 = reference_simulation_1.calculate('revdisp')
reform_revdisp_1 = reform_simulation_1.calculate('revdisp')

display('reference value', reference_revdisp_1, 'reform value', reform_revdisp_1)

from openfisca_france.model.common import revenu_net_individu

class revenu_net_individu_2(formulas.SimpleFormulaColumn):
    column = columns.FloatCol
    entity_class = entities.Individus
    label = u"Revenu net de l'individu"

    def function(self, simulation, period):
        period = period.start.offset('first-of', 'year').period('year')
        rev_trav = simulation.calculate('rev_trav', period)
        return period, rev_trav

def build_reform_2(tax_benefit_system):
    reform_entity_class_by_key_plural = reforms.clone_entity_classes(entities.entity_class_by_key_plural)

    # Change the ReformIndividus entity "revenu_net_individu" variable.
    ReformIndividus = reform_entity_class_by_key_plural['individus']
    ReformIndividus.column_by_name['revenu_net_individu'] = revenu_net_individu_2

    return reforms.Reform(
        entity_class_by_key_plural = reform_entity_class_by_key_plural,
        legislation_json = tax_benefit_system.legislation_json,  # Keep the reference legislation.
        name = u'Dummy reform 2 (formula only)',
        reference = tax_benefit_system,
        )

dummy_reform_2 = build_reform_2(tax_benefit_system)  # tax_benefit_system was initialized above.

scenario_2 = dummy_reform_2.new_scenario().init_single_entity(
    period = 2014,
    parent1 = dict(
        birth = datetime.date(1974, 1, 1),
        sali = 5000,
        ),
    )

reference_simulation_2 = scenario_2.new_simulation(reference=True)
reform_simulation_2 = scenario_2.new_simulation()

# Calculate the "Revenu disponible" for both reference and reform simulations.
reference_revdisp_2 = reference_simulation_2.calculate('revdisp')
reform_revdisp_2 = reform_simulation_2.calculate('revdisp')

display('reference value', reference_revdisp_2, 'reform value', reform_revdisp_2)

def build_reform(tax_benefit_system):
    reform_entity_class_by_key_plural = reforms.clone_entity_classes(entities.entity_class_by_key_plural)

    # Change the ReformIndividus entity "revenu_net_individu" variable.
    ReformIndividus = reform_entity_class_by_key_plural['individus']
    ReformIndividus.column_by_name['revenu_net_individu'] = revenu_net_individu_2

    return reforms.Reform(
        entity_class_by_key_plural = reform_entity_class_by_key_plural,
        legislation_json = reform_legislation_json,  # Was built from the XML file you uploaded.
        name = u'Dummy reform',
        reference = tax_benefit_system,
        )

dummy_reform = build_reform(tax_benefit_system)  # tax_benefit_system was initialized above.

scenario = dummy_reform.new_scenario().init_single_entity(
    period = 2014,
    parent1 = dict(
        birth = datetime.date(1974, 1, 1),
        sali = 5000,
        ),
    )

reference_simulation = scenario.new_simulation(reference=True)
reform_simulation = scenario.new_simulation()

# Calculate the "Revenu disponible" for both reference and reform simulations.
reference_revdisp = reference_simulation.calculate('revdisp')
reform_revdisp = reform_simulation.calculate('revdisp')

display('reference value', reference_revdisp, 'reform value', reform_revdisp)



