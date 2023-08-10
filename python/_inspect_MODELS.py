from milsed.models import MODELS
import pickle

pumpfile = '/home/js7561/dev/milsed/models/resources/pump.pkl'
with open(pumpfile, 'rb') as fp:
    pump = pickle.load(fp)

alpha = 1.0

model, inputs, outputs = MODELS['crnn1d_smp'](pump, alpha)

model.summary()

model, inputs, outputs = MODELS['crnn1d_max'](pump, alpha)

model.summary()

model, inputs, outputs = MODELS['crnn1d_avg'](pump, alpha)

model.summary()

model, inputs, outputs = MODELS['cnn1d_smp'](pump, alpha)

model.summary()



