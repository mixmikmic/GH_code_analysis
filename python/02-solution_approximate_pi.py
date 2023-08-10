import numpy as np
import numpy.testing as nptest

def create_trial_chain(N,max_step_size):
    if(type(N) == int and N > 0):#Checking if first argument is valid
        if((type(max_step_size) == float or type(max_step_size)== int) # Checking if second argument is valid
           and max_step_size>=0 and max_step_size <=1):
            # initialize with -1 to make shure that all values will be changed
            markov_chain = np.ones((N,2))*(-1) 
            #set a random start_point
            markov_chain[0,:]=np.random.rand(2)

            for i in range(1,N):
                previous = markov_chain[i-1,:]
                delta = (np.random.rand(2)-0.5)*max_step_size*2 
                # if max_step_size is 1 this is a random number from [-1,1]
                proposal = previous+delta
                if ((proposal <= 1)*(proposal >= 0)).all():
                    markov_chain[i,:]=proposal
                else:
                    markov_chain[i,:]=previous
                    
            return markov_chain
        else:
            raise TypeError("max_step_size is not a float between 0 and 1")
    else:
        raise TypeError("N is not a positive integer")

def approx_pi_markov(N,max_step_size):
    trial_chain = create_trial_chain(N,max_step_size)
    return 4/N *np.sum(np.power(trial_chain[:,0],2) + np.power(trial_chain[:,1],2) <= 1)

get_ipython().magic('load_ext ipython_nose')

get_ipython().run_cell_magic('nose', '', "def test_create_trial_chain_returns_nx2_array_with_positive_input():\n    nptest.assert_equal(np.shape(create_trial_chain(3,1)),(3,2))\n    \ndef test_do_not_create_trial_chain_with_0_input():\n    with nptest.assert_raises(TypeError):\n        create_trial_chain(0,1)\n    \ndef test_do_not_create_trial_chain_with_negative_input():\n    with nptest.assert_raises(TypeError):\n        create_trial_chain(-3,1)\n    \ndef test_do_not_create_trial_chain_with_float():\n    with nptest.assert_raises(TypeError):\n        create_trial_chain(3.3,1)\n        \ndef test_do_not_create_trial_chain_with_string_as_second_arg():\n    with nptest.assert_raises(TypeError):\n        create_trial_chain(3,'foo')\n        \ndef test_do_not_create_trial_chain_with_second_arg_larger_one():\n    with nptest.assert_raises(TypeError):\n        create_trial_chain(3,1.4)\n\ndef test_do_not_create_trial_chain_with_second_arg_smaller_0():\n    with nptest.assert_raises(TypeError):\n        create_trial_chain(3,-1)\n\ndef test_create_trial_chain_returns_numbers_between_0_and_1():\n    for i in range(50):\n        test_array = create_trial_chain(1000,0.5)\n        assert ((test_array >=0) * (test_array <= 1)).all()\n\ndef test_do_not_app_pi_with_second_arg_smaller_0():\n    with nptest.assert_raises(TypeError):\n        approx_pi_markov(10000,-1)\n        \n\ndef test_approx_pi_markov_is_close_to_py():\n    for i in range(20):\n        assert abs(approx_pi_markov(10000,0.5)-np.pi) <= 0.1")

