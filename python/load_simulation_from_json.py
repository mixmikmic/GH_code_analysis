get_ipython().magic('pylab nbagg')

import os
import json
from tvb.simulator.lab import *
LOG = get_logger('demo')
from tvb.datatypes import noise_framework
from tvb.datatypes.equations import HRFKernelEquation
from tvb.basic.traits.parameters_factory import *

available_models = get_traited_subclasses(models.Model)
available_monitors = get_traited_subclasses(monitors.Monitor)
available_integrators = get_traited_subclasses(integrators.Integrator)
available_couplings = get_traited_subclasses(coupling.Coupling)

with open('tvb_simulation_3.json', 'r') as fdr:
    with open('tvbsim.json', 'w') as fdw:
        json.dump(json.loads(json.load(fdr)['_simulator_configuration']), fdw, indent=1)

json_path = "tvb_simulation_3.json"

def ignore_key(key): # v1.4 -> 1.5 compat
    keys = ['_noise', 'pre_expr', 'post_expr']
    return any(k in key for k in keys)

with open(json_path, 'rb') as input_file:
    simulation_json = input_file.read()
    simulation_json = json.loads(simulation_json)
    simulation_config = {}
    for key, val in json.loads(simulation_json["_simulator_configuration"]):
        nonempty_key = len(key) > 1
        no_noise_v14 = '_noise' not in key
        no_pre_expr_v141 = key.startswith('monitors') and not key.endswith('pre_expr')
        no_post_expr_v141 = key.startswith('monitors') and not key.endswith('post_expr')
        if nonempty_key and no_pre_expr_v141 and no_post_expr_v141:
            simulation_config[key, valu]

model_key = 'model'
model_name = simulation_config[model_key]
#noise_key = '%s%s%s%s_noise' % (model_key, KEYWORD_PARAMS, KEYWORD_OPTION, model_name)
#noise_name = simulation_config[noise_key]
#random_stream_key = '%s%s%s%s_random_stream' % (noise_key, KEYWORD_PARAMS, KEYWORD_OPTION, noise_name)

selectors = ['coupling', 'integrator']
selectors.append(model_key)
#selectors.append(noise_key)
#selectors.append(random_stream_key)
                                    
transformed = collapse_params(simulation_config, selectors)

converted = {str(k): try_parse(v) for k,v in transformed.iteritems() }

model_parameters = converted['model_parameters']
integrator_parameters = converted['integrator_parameters']
coupling_parameters = converted['coupling_parameters']

# TODO: this parameter shuld be correctly parsed and considered:
del model_parameters['state_variable_range_parameters']

noise_framework.build_noise(model_parameters)
noise_framework.build_noise(integrator_parameters)

model_instance = available_models[converted['model']](**model_parameters)
integr_instance = available_integrators[converted['integrator']](**integrator_parameters)
coupling_inst = available_couplings[converted['coupling']](**coupling_parameters)
conduction_speed = converted['conduction_speed']

# TODO: reloading the original Connectivity ...
# TODO: detect surface simulations and reconfigure ...
conn = connectivity.Connectivity(load_default=True)
model_instance

monitors = converted['monitors']
monitors_parameters = converted['monitors_parameters']
monitors_list = []

for monitor_name in monitors:
    if (monitors_parameters is not None) and (str(monitor_name) in monitors_parameters):
        current_monitor_parameters = monitors_parameters[str(monitor_name)]
        HRFKernelEquation.build_equation_from_dict('hrf_kernel', current_monitor_parameters, True)
        monitors_list.append(available_monitors[str(monitor_name)](**current_monitor_parameters))
    else:
        ### We have monitors without any UI settable parameter.
        monitors_list.append(available_monitors[str(monitor_name)]())

sim = simulator.Simulator(connectivity=conn, coupling=coupling_inst, 
                          model=model_instance, integrator=integr_instance,
                          monitors=monitors_list, conduction_speed=conduction_speed)
sim.configure()

result_data = {m_name: [] for m_name in monitors}
result_time = {m_name: [] for m_name in monitors}

simulation_length = converted['simulation_length']

for result in sim(simulation_length=simulation_length):
    for j, monitor_name in enumerate(monitors):
        if result[j] is not None:
            result_time[monitor_name].append(result[j][0])
            result_data[monitor_name].append(result[j][1])

for j, monitor_name in enumerate(monitors):
    figure(j + 1)
    mon_data = numpy.array(result_data[monitor_name])
    for sv in xrange(mon_data.shape[1]):
        for m in xrange(mon_data.shape[3]):
            plot(result_time[monitor_name], mon_data[:, 0, :, 0])
            title(monitor_name + " -- State variable " + str(sv) + " -- Mode " + str(m))

