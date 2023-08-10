# Please modify the below as you see fit
resource_group = "<RESOURCE_GROUP>" 
aks_name = "<AKS_CLUSTER_NAME>"
location = "eastus"

image_name = '<YOUR_DOCKER_IMAGE>' # 'masalvar/tfresnet-gpu' Feel free to use this Image if you want to 
                                   # skip creating your own container
selected_subscription = "'<YOUR SUBSCRIPTION>'" # If you have multiple subscriptions select 
                                                # the subscription you want to use here

get_ipython().system('az login -o table')

get_ipython().system('az account set --subscription $selected_subscription')

get_ipython().system('az account show')

get_ipython().system('az provider register -n Microsoft.ContainerService')

get_ipython().system('az group create --name $resource_group --location $location')

get_ipython().system('az aks create --resource-group $resource_group --name $aks_name --node-count 1 --generate-ssh-keys -s Standard_NC6')

get_ipython().system('sudo az aks install-cli')

get_ipython().system('az aks get-credentials --resource-group $resource_group --name $aks_name')

get_ipython().system('kubectl get nodes')

get_ipython().system('kubectl get pods --all-namespaces')

app_template = {
  "apiVersion": "apps/v1beta1",
  "kind": "Deployment",
  "metadata": {
      "name": "azure-dl"
  },
  "spec":{
      "replicas":1,
      "template":{
          "metadata":{
              "labels":{
                  "app":"azure-dl"
              }
          },
          "spec":{
              "containers":[
                  {
                      "name": "azure-dl",
                      "image": image_name,
                      "env":[
                          {
                              "name": "LD_LIBRARY_PATH",
                              "value": "$LD_LIBRARY_PATH:/usr/local/nvidia/lib64:/opt/conda/envs/py3.6/lib"
                          }
                      ],
                      "ports":[
                          {
                              "containerPort":80,
                              "name":"model"
                          }
                      ],
                      "volumeMounts":[
                          {
                            "mountPath": "/usr/local/nvidia",
                            "name": "nvidia"
                          }
                      ],
                      "resources":{
                           "requests":{
                               "alpha.kubernetes.io/nvidia-gpu": 1
                           },
                           "limits":{
                               "alpha.kubernetes.io/nvidia-gpu": 1
                           }
                       }  
                  }
              ],
              "volumes":[
                  {
                      "name": "nvidia",
                      "hostPath":{
                          "path":"/usr/local/nvidia"
                      },
                  },
              ]
          }
      }
  }
}

service_temp = {
  "apiVersion": "v1",
  "kind": "Service",
  "metadata": {
      "name": "azure-dl"
  },
  "spec":{
      "type": "LoadBalancer",
      "ports":[
          {
              "port":80
          }
      ],
      "selector":{
            "app":"azure-dl"
      }
   }
}

def write_json_to_file(json_dict, filename, mode='w'):
    with open(filename, mode) as outfile:
        json.dump(json_dict, outfile, indent=4,sort_keys=True)
        outfile.write('\n\n')

write_json_to_file(app_template, 'az-dl.json') # We write the service template to the json file

write_json_to_file(service_temp, 'az-dl.json', mode='a') # We add the loadbelanacer template to the json file

get_ipython().system('cat az-dl.json')

get_ipython().system('kubectl create -f az-dl.json')

get_ipython().system('kubectl get pods --all-namespaces')

get_ipython().system('kubectl get events')

pod_json = get_ipython().getoutput('kubectl get pods -o json')
pod_dict = json.loads(''.join(pod_json))
get_ipython().system("kubectl logs {pod_dict['items'][0]['metadata']['name']}")

get_ipython().system('kubectl get service azure-dl')

get_ipython().system('kubectl delete -f az-dl.json')

get_ipython().system('az aks delete -n $aks_name -g $resource_group -y')

get_ipython().system('az group delete --name $resource_group -y')

