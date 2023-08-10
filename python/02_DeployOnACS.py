resource_group = "m*****g" # Feel free to modify these
acs_name = "m*****s"
location = "s*****s"

image_name = 'masalvar/cntkresnet' 
selected_subscription = "'Azure *****'" # If you have multiple subscriptions select 
                                        # the subscription you want to use here

get_ipython().system('az login -o table')

get_ipython().system('az account set --subscription $selected_subscription')

get_ipython().system('az account show')

get_ipython().system('az group create --name $resource_group --location $location')

import os
if not os.path.exists('{}/.ssh/id_rsa'.format(os.environ['HOME'])):
    get_ipython().system('ssh-keygen -t rsa -b 2048 -N "" -f ~/.ssh/id_rsa')

json_data = get_ipython().getoutput('az acs create --name $acs_name --resource-group $resource_group --admin-username mat --dns-prefix $acs_name --agent-count 2')

json_dict = json.loads(''.join(json_data))

if json_dict['properties']['provisioningState'] == 'Succeeded':
    print('Succensfully provisioned ACS {}'.format(acs_name))
    _,ssh_addr,_,_,ssh_port, = json_dict['properties']['outputs']['sshMaster0']['value'].split()

get_ipython().system('az acs list --resource-group $resource_group --output table')

get_ipython().system('az acs show --name $acs_name --resource-group $resource_group')

get_ipython().run_cell_magic('bash', '--bg -s $ssh_port $ssh_addr', 'ssh -o StrictHostKeyChecking=no -fNL 1212:localhost:80 -p $1 $2')

application_id = "/cntkresnet"

app_template = {
  "id": application_id,
  "cmd": None,
  "cpus": 1,
  "mem": 1024,
  "disk": 100,
  "instances": 1,
  "acceptedResourceRoles": [
    "slave_public"
  ],
  "container": {
    "type": "DOCKER",
    "volumes": [],
    "docker": {
      "image": image_name,
      "network": "BRIDGE",
      "portMappings": [
        {
          "containerPort": 88,
          "hostPort": 80,
          "protocol": "tcp",
          "name": "80",
          "labels": {}
        }
      ],
      "privileged": False,
      "parameters": [],
      "forcePullImage": True
    }
  },
  "healthChecks": [
    {
      "path": "/",
      "protocol": "HTTP",
      "portIndex": 0,
      "gracePeriodSeconds": 300,
      "intervalSeconds": 60,
      "timeoutSeconds": 20,
      "maxConsecutiveFailures": 3
    }
  ]
}

def write_json_to_file(json_dict, filename):
    with open(filename, 'w') as outfile:
        json.dump(json_dict, outfile)

write_json_to_file(app_template, 'marathon.json')

get_ipython().system('curl -X POST http://localhost:1212/marathon/v2/apps -d @marathon.json -H "Content-type: application/json"')

from time import sleep
for i in range(20):
    json_data = get_ipython().getoutput('curl http://localhost:1212/marathon/v2/apps')
    if json.loads(json_data[-1])['apps'][0]['tasksRunning']==1:
        print('Web app ready')
        break
    else:
        print('Preparing Web app')
    sleep(10)
else:
    print('Timeout! Something went wrong!')

app_url = json_dict['properties']['outputs']['agentFQDN']['value']

print('Application URL: {}'.format(app_url))
print('Application ID: {}'.format(application_id))

get_ipython().system('az acs delete --resource-group $resource_group --name $acs_name')

get_ipython().system('az group delete --name $resource_group -y')

