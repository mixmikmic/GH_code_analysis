get_ipython().system('kubectl config set-context $(kubectl config current-context) --namespace=kubeflow-seldon')

get_ipython().system('python -m grpc.tools.protoc -I. --python_out=. --grpc_python_out=. ./proto/prediction.proto')

get_ipython().run_line_magic('matplotlib', 'inline')
import utils
from visualizer import get_graph
mnist = utils.download_mnist()

get_graph("../k8s_serving/serving_model.json",'r')

get_ipython().system('pygmentize ../k8s_serving/serving_model.json')

get_ipython().system('kubectl apply -f ../k8s_serving/serving_model.json')

get_ipython().system("kubectl get seldondeployments mnist-classifier -o jsonpath='{.status}'")

utils.predict_rest_mnist(mnist)

utils.predict_grpc_mnist(mnist)

get_ipython().system("kubectl label nodes $(kubectl get nodes -o jsonpath='{.items[0].metadata.name}') role=locust")

get_ipython().system('helm install seldon-core-loadtesting --name loadtest      --namespace kubeflow-seldon     --repo https://storage.googleapis.com/seldon-charts     --set locust.script=mnist_rest_locust.py     --set locust.host=http://mnist-classifier:8000     --set oauth.enabled=false     --set oauth.key=oauth-key     --set oauth.secret=oauth-secret     --set locust.hatchRate=1     --set locust.clients=1     --set loadtest.sendFeedback=1     --set locust.minWait=0     --set locust.maxWait=0     --set replicaCount=1     --set data.size=784')

get_graph("../k8s_serving/ab_test_sklearn_tensorflow.json",'r')

get_ipython().system('pygmentize ../k8s_serving/ab_test_sklearn_tensorflow.json')

get_ipython().system('kubectl apply -f ../k8s_serving/ab_test_sklearn_tensorflow.json')

get_ipython().system("kubectl get seldondeployments mnist-classifier -o jsonpath='{.status}'")

utils.predict_rest_mnist(mnist)

utils.evaluate_abtest(mnist,100)

get_graph("../k8s_serving/epsilon_greedy_3way.json",'r')

get_ipython().system('pygmentize ../k8s_serving/epsilon_greedy_3way.json')

get_ipython().system('kubectl apply -f ../k8s_serving/epsilon_greedy_3way.json')

get_ipython().system("kubectl get seldondeployments mnist-classifier -o jsonpath='{.status}'")

utils.predict_rest_mnist(mnist)

utils.evaluate_egreedy(mnist,100)

get_ipython().system('helm delete loadtest --purge')



