import dopapy
import dopapy.trading as dp
import dopapy.types as dd
from dopapy.core.services.instance.iservice import ILabeledLearningServiceInstance
import numpy as np
import os
import warnings
from mnist import MnistData

warnings.filterwarnings('ignore') #Thats just to ignore warnings that are irrelevant for this type of sample

class Teacher(object):
    def __init__(self, port, from_sample, to_sample, name):
        mnist_data = MnistData(from_sample,to_sample)        
        self.x_train = mnist_data.x_train
        self.y_train = mnist_data.y_train
        self.name = name
        self.port = port
        
    def teach(self):
        print("Teacher {0} started teaching.".format(self.name))
        session = dp.create_session(master_password='password123456',
                                    keystore_dir='/data/dopamine/keystore_dir_a/',
                                    web_server_host="127.0.0.1",
                                    web_server_port=self.port,
                                    dopmn_budget=10000,
                                    ether_budget=int(10E18))
        counter_quote = dp.get_counter_quote(url="https://127.0.0.1:8007", quote_id=0)
        my_service_descriptor = dd.ServiceDescriptor(
            service_role=dd.ServiceRole.CONSUMER,
            input_descriptors=[dd.TensorDescriptor([-1,28,28,1])],
            output_descriptors=[dd.TensorDescriptor([-1])],
            service_type=dd.ServiceType.Learning.LABELED,
            remote_sources=[counter_quote]
        )
        reward_desc = dp.create_reward_descriptor(max_dopamine_price=10)
        reward_desc.add_payment(dd.RewardPaymentType.UPDATE, max_price=20)
        reward_desc.add_payment(dd.RewardPaymentType.FORWARD, max_price=0)
        quote = dp.create_quote_services(
            session=session,
            service_descriptor=my_service_descriptor,
            side=dp.BUY,
            reward_descriptor=reward_desc)
        student = quote.get_service()        
        student.train(input_objects= [dd.Tensor(self.x_train)], labels= [dd.Tensor(self.y_train)])
        print("Done")

teachers = [Teacher(8008, 0,     10000, "A")
,Teacher(8009, 10001, 30000, "B")
,Teacher(8011, 30001, 50000, "C")
,Teacher(8012, 50001, 55000, "D")
,Teacher(8013, 55001, 60000, "E")]

for t in teachers:
    t.teach()



