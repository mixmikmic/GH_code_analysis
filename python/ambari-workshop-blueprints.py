get_ipython().run_cell_magic('bash', '', '\n## Upload the Blueprint\n\nambari_server=pregion-sean01.cloud.hortonworks.com ## This is the hostname of the ambari-server.\n                       ##   - Set to localhost if running directly on the server.\npass=admin        ## The Ambari admin password\n\ncurl -u admin:${pass} \\\n  -H X-Requested-By:seano \\\n  -X POST -d @blueprints/hdp-201503_blueprint.json \\\n  http://${ambari_server}:8080/api/v1/blueprints/hdp-201503')

get_ipython().run_cell_magic('bash', '', '\n## Create the cluster\n\nambari_server=pregion-sean01.cloud.hortonworks.com\npass=admin\n\ncurl -u admin:${pass} \\\n  -H X-Requested-By:seano \\\n  -X POST \\\n  -d @blueprints/hdp-201503_cluster.json \\\n  http://${ambari_server}:8080/api/v1/clusters/mycluster')

get_ipython().run_cell_magic('bash', '', '\nambari_server=hostname\npass=mypassword\n\ncurl -u admin:${pass} http://${ambari_server}:8080/api/v1/clusters/mycluster/requests/1')

get_ipython().run_cell_magic('bash', '', '\n# change this to fit your configuration \nambari_server=privateIPorHostnameOfTheServer\n\n## the URL to the installer script\nbootstrap_url=https://raw.githubusercontent.com/seanorama/ambari-bootstrap/master/ambari-bootstrap.sh\n \n## install the ambari-server\npdsh -w ${ambari_server} "curl -sSL ${bootstrap_url} | install_ambari_server=true sh"\n\n## install to all other nodes. See ‘man pdsh’ for the various ways you can specify hosts.\npdsh -w ${stack}0[2-3].cloud.hortonworks.com "curl -sSL ${bootstrap_url} | ambari_server=${ambari_server} sh"')



