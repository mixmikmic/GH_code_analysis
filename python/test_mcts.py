#%load_ext autoreload
#%autoreload 2
#%reload_ext autoreload

from concept_dependency_graph import ConceptDependencyGraph
import data_generator as dg
from student import *
import simple_mdp as sm

n_concepts = 4
use_student2 = True
student2_str = '2' if use_student2 else ''
learn_prob = 0.15
lp_str = '-lp{}'.format(int(learn_prob*100)) if not use_student2 else ''
n_students = 100000
seqlen = 6
filter_mastery = False
filter_str = '' if not filter_mastery else '-filtered'
policy = 'random'
filename = 'test{}-n{}-l{}{}-{}{}.pickle'.format(student2_str, n_students, seqlen,
                                                    lp_str, policy, filter_str)
#concept_tree = sm.create_custom_dependency()
concept_tree = ConceptDependencyGraph()
concept_tree.init_default_tree(n_concepts)
if not use_student2:
    test_student = Student(n=n_concepts,p_trans_satisfied=learn_prob, p_trans_not_satisfied=0.0, p_get_ex_correct_if_concepts_learned=1.0)
else:
    test_student = Student2(n_concepts)
print(filename)

print ("Initializing synthetic data sets...")
dg.generate_data(concept_tree, student=test_student, n_students=n_students, filter_mastery=filter_mastery, seqlen=seqlen, policy=policy, filename="{}{}".format(dg.SYN_DATA_DIR, filename))
print ("Data generation completed. ")

import dynamics_model_class as dmc
import numpy as np
import dataset_utils

# test_model hidden=16
# test_model_mid hidden=10
# test_model_small hidden=5
# test_model_tiny hidden=3
dmodel = dmc.DynamicsModel(model_id="test2_model_mid", timesteps=seqlen, load_checkpoint=False)

# load toy data
data = dataset_utils.load_data(filename='{}{}'.format(dg.SYN_DATA_DIR, filename))
print('Average posttest: {}'.format(sm.expected_reward(data)))
print('Percent of full posttest score: {}'.format(sm.percent_complete(data)))
print('Percent of all seen: {}'.format(sm.percent_all_seen(data)))
input_data_, output_mask_, target_data_ = dataset_utils.preprocess_data_for_rnn(data)

train_data = (input_data_[:,:,:], output_mask_[:,:,:], target_data_[:,:,:])
print(input_data_.shape)
print(output_mask_.shape)
print(target_data_.shape)

dmodel.train(train_data, n_epoch=50, load_checkpoint=False)

preds = dmodel.predict(input_data_[:1,:2, :])
print len(preds)
print len(preds[0])
print preds[0]

rnnsim = dmc.RnnStudentSim(dmodel)

from mcts import *
actix = 0
actvec = np.zeros((n_concepts,))
actvec[actix] = 1
act = StudentAction(actix, actvec)
print rnnsim.sequence
print rnnsim.sample_observations()

rnnsim.advance_simulator(act, 1)

print rnnsim.sequence
print rnnsim.sample_observations()
print rnnsim.sample_reward()



