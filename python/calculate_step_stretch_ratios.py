# Imports
get_ipython().magic('matplotlib inline')
import pardir; pardir.pardir() # Allow imports from parent directory
import fibonaccistretch as fib
import bjorklund

# Setting up basics
original_rhythm = [1,0,0,1,0,0,1,0]
target_rhythm = [1,0,0,0,0,1,0,0,0,0,1,0,0]

fib.calculate_pulse_ratios(original_rhythm, target_rhythm)

fib.calculate_pulse_lengths(original_rhythm)

fib.calculate_pulse_ratios([1]*len(original_rhythm), target_rhythm)

[1]*8

fib.calculate_pulse_lengths(target_rhythm)

# Original and target pulse lengths; just take the first one from each for now
opl = fib.calculate_pulse_lengths(original_rhythm)[0]
tpl = fib.calculate_pulse_lengths(target_rhythm)[0]

# Adapted from Euclidean stretch
opr = [1] * len(original_rhythm)

# Generate target pulse rhythm ("tpr")
tpr = bjorklund.bjorklund(pulses=opl, steps=tpl)
tpr_pulse_lengths = fib.calculate_pulse_lengths(tpr)
tpr_pulse_ratios = fib.calculate_pulse_ratios(opr, tpr)

tpr_pulse_ratios

# Format pulse ratios so there's one for each step
original_pulse_lengths = fib.calculate_pulse_lengths(original_rhythm)
pulse_ratios = fib.calculate_pulse_ratios(original_rhythm, target_rhythm)
pulse_ratios_by_step = []
for i,pulse_length in enumerate(original_pulse_lengths):
    for _ in range(pulse_length):
        pulse_ratios_by_step.append(pulse_ratios[i])
pulse_ratios_by_step

def calculate_step_stretch_ratios(original_rhythm, target_rhythm):
    # Original and target pulse lengths
    original_pulse_lengths = fib.calculate_pulse_lengths(original_rhythm)
    target_pulse_lengths = fib.calculate_pulse_lengths(target_rhythm)

    # Pulse ratios
    # Format pulse ratios so there's one for each step
    pulse_ratios = fib.calculate_pulse_ratios(original_rhythm, target_rhythm)
    pulse_ratios_by_step = []
    for i,pulse_length in enumerate(original_pulse_lengths):
        for _ in range(pulse_length):
            pulse_ratios_by_step.append(pulse_ratios[i])

    # Calculate stretch ratios for each original step
    # Adapted from Euclidean stretch
    step_stretch_ratios = []
    for i in range(min(len(original_pulse_lengths), len(target_pulse_lengths))):
        # Pulse lengths
        opl = original_pulse_lengths[i]
        tpl = target_pulse_lengths[i]

        # Use steps as original pulse rhythm ("opr")
        opr = [1] * len(original_rhythm)

        # Generate target pulse rhythm ("tpr") using Bjorklund's algorithm
        tpr = bjorklund.bjorklund(pulses=opl, steps=tpl)
        tpr_pulse_lengths = fib.calculate_pulse_lengths(tpr)
        tpr_pulse_ratios = fib.calculate_pulse_ratios(opr, tpr)

        # Scale the tpr pulse ratios by the corresponding ratio from pulse_ratios_by_step
        tpr_pulse_ratios *= pulse_ratios_by_step[i]

        step_stretch_ratios.extend(tpr_pulse_ratios)
    
    return step_stretch_ratios

step_stretch_ratios = calculate_step_stretch_ratios(original_rhythm, target_rhythm)
step_stretch_ratios

sum(step_stretch_ratios) / len(original_rhythm)

step_stretch_ratios = calculate_step_stretch_ratios(original_rhythm, original_rhythm)
step_stretch_ratios

sum(step_stretch_ratios) / len(original_rhythm)



stretch_multiplier = 1.0 / (sum(step_stretch_ratios) / len(original_rhythm))
stretch_multiplier

step_stretch_ratios = [r * stretch_multiplier for r in step_stretch_ratios]
step_stretch_ratios

sum(step_stretch_ratios) / len(original_rhythm)

def calculate_step_stretch_ratios(original_rhythm, target_rhythm):
    # Original and target pulse lengths
    original_pulse_lengths = list(fib.calculate_pulse_lengths(original_rhythm))
    target_pulse_lengths = list(fib.calculate_pulse_lengths(target_rhythm))

    # Pulse ratios
    # Format pulse ratios so there's one for each step
    pulse_ratios = list(fib.calculate_pulse_ratios(original_rhythm, target_rhythm))
    if len(pulse_ratios) < len(original_pulse_lengths):  # Add 0s to pulse ratios if there aren't enough
        for _ in range(len(original_pulse_lengths) - len(pulse_ratios)):
            pulse_ratios.append(0.0)
    assert(len(pulse_ratios) == len(original_pulse_lengths))
    pulse_ratios_by_step = []
    for i,pulse_length in enumerate(original_pulse_lengths):
        for _ in range(pulse_length):
            pulse_ratios_by_step.append(pulse_ratios[i])

    # Calculate stretch ratios for each original step
    # Adapted from Euclidean stretch
    step_stretch_ratios = []
    for i in range(min(len(original_pulse_lengths), len(target_pulse_lengths))):
        # Pulse lengths
        opl = original_pulse_lengths[i]
        tpl = target_pulse_lengths[i]
        
        # Adjust target pulse length if it's too small
        #if opl > tpl:
        #    tpl = opl
        while opl > tpl:
           tpl *= 2

        # Use steps as original pulse rhythm ("opr")
        opr = [1] * len(original_rhythm)

        # Generate target pulse rhythm ("tpr") using Bjorklund's algorithm
        tpr = bjorklund.bjorklund(pulses=opl, steps=tpl)
        tpr_pulse_lengths = fib.calculate_pulse_lengths(tpr)
        tpr_pulse_ratios = fib.calculate_pulse_ratios(opr, tpr)

        # Scale the tpr pulse ratios by the corresponding ratio from pulse_ratios_by_step
        tpr_pulse_ratios *= pulse_ratios_by_step[i]

        step_stretch_ratios.extend(tpr_pulse_ratios)
        
    # Multiply by stretch multiplier to make sure the length is the same as original
    stretch_multiplier = 1.0 / (sum(step_stretch_ratios) / len(original_rhythm))
    step_stretch_ratios = [r * stretch_multiplier for r in step_stretch_ratios]
    assert(round(sum(step_stretch_ratios) / len(original_rhythm), 5) == 1)  # Make sure it's *close enough* to original length.
    
    return step_stretch_ratios

step_stretch_ratios = calculate_step_stretch_ratios(original_rhythm, target_rhythm)
step_stretch_ratios

calculate_step_stretch_ratios(original_rhythm, [1,0,1])
# fib.calculate_pulse_ratios(original_rhythm, [1,0,1])

get_ipython().magic('pinfo round')

reload(fib)

# import numpy as np
# a = np.array(original_rhythm)
# b = np.zeros(4)
# np.hstack((a, b))

fib.calculate_step_stretch_ratios(original_rhythm, target_rhythm)



