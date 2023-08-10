MODEL_NAME = 'class-model-01'
model_dir = 'trained_models/{}'.format(MODEL_NAME)
print(model_dir)

from google.datalab.ml import TensorBoard
TensorBoard().start(model_dir)
TensorBoard().list()

# to stop TensorBoard
TensorBoard().stop(23002)
print('stopped TensorBoard')
TensorBoard().list()

