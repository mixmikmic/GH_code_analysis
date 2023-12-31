def first(step, badstep=None):
    # Increment the step 
    step += 1 
    
    # Check if this is a bad step 
    if badstep == step:
        raise ValueError("Failed after {} steps".format(step))
    
    # Call sub steps in order 
    step = first_task_one(step, badstep)
    step = first_task_two(step, badstep)
    
    # Return the step that we're on 
    return step 


def first_task_one(step, badstep=None):
    # Increment the step 
    step += 1 
    
    # Check if this is a bad step 
    if badstep == step:
        raise ValueError("Failed after {} steps".format(step))
    
    # Call sub steps in order 
    step = first_task_one_subtask_one(step, badstep)
    
    # Return the step that we're on 
    return step 


def first_task_one_subtask_one(step, badstep=None):
    # Increment the step 
    step += 1 
    
    # Check if this is a bad step 
    if badstep == step:
        raise ValueError("Failed after {} steps".format(step))
    
    # Return the step that we're on 
    return step 


def first_task_two(step, badstep=None):
    # Increment the step 
    step += 1 
    
    # Check if this is a bad step 
    if badstep == step:
        raise ValueError("Failed after {} steps".format(step))
    
    # Return the step that we're on 
    return step 


def second(step, badstep=None):
    # Increment the step 
    step += 1 
    
    # Check if this is a bad step 
    if badstep == step:
        raise ValueError("Failed after {} steps".format(step))
    
    # Call sub steps in order 
    step = second_task_one(step, badstep)
    
    # Return the step that we're on 
    return step 


def second_task_one(step, badstep=None):
    # Increment the step 
    step += 1 
    
    # Check if this is a bad step 
    if badstep == step:
        raise ValueError("Failed after {} steps".format(step))
    
    # Return the step that we're on 
    return step 

def main(badstep=None, **kwargs):
    """
    This function is the entry point of the program, it does 
    work on the arguments by calling each step function, which 
    in turn call substep functions. 
    
    Passing in a number for badstep will cause whichever step 
    that is to raise an exception. 
    """
    
    step = 0 # count the steps 
    
    # Execute each step one at a time. 
    step = first(step, badstep) 
    step = second(step, badstep)
    
    # Return a report 
    return "Sucessfully executed {} steps".format(step) 

if __name__ == "__main__":
    main()

main(3)

import random 

class RandomError(Exception):
    """
    A custom exception for this code block. 
    """
    pass 


def randomly_errors(p_error=0.5):
    if random.random() <= p_error:
        raise RandomError("Error raised with {:0.2f} likelihood!".format(p_error))


try:
    randomly_errors(0.5) 
    print("No error occurred!")
except RandomError as e:
    print(e)
finally:
    print("This runs no matter what!")

