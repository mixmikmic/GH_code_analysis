get_ipython().run_line_magic('load_ext', 'sparkmagic.magics')

get_ipython().run_cell_magic('spark', 'config ', '{\n  "name":"remote_eth_producer",\n  "driverMemory":"1G",\n  "numExecutors":1,\n  "proxyUser":"noobie",\n  "archives": ["hdfs:///user/noobie/gethdemo.tar.gz"],\n  "files" : ["hdfs:///user/noobie/noobie.keytab"],\n  "queue": "streaming",\n  "conf": {"spark.yarn.appMasterEnv.PYSPARK_PYTHON":"gethdemo.tar.gz/demo/bin/python3.5",\n          "PYSPARK_PYTHON":"gethdemo.tar.gz/demo/bin/python3.5"\n          }\n}')

get_ipython().run_line_magic('spark', 'add -s ethlogproducer -l python -u http://livy.server.url:8999 --auth Kerberos')

get_ipython().run_cell_magic('spark', '', 'import subprocess\nkinit = \'/usr/bin/kinit\'\nkinit_args = [kinit, \'-kt\', "noobie.keytab" , "noobie"]\nsubprocess.check_output(kinit_args)')

get_ipython().run_cell_magic('spark', '-s ethlogproducer', 'from web3 import Web3, HTTPProvider, IPCProvider\n\ngethRPCUrl=\'http://local.geth.address:8545\'\nweb3 = Web3(HTTPProvider(gethRPCUrl))\n\n# Retrieve the last block number available from geth RPC\ncurrentblock = web3.eth.getBlock(\'latest\').number\nprint("Latest block: " + str(currentblock))')

get_ipython().run_cell_magic('spark', '-s ethlogproducer', 'from hexbytes import HexBytes\nimport threading, logging, time, json\n\nclass HexJsonEncoder(json.JSONEncoder):\n    def default(self, obj):\n        if isinstance(obj, HexBytes):\n            return obj.hex()\n        return super().default(obj)')

get_ipython().run_cell_magic('spark', '', 'from kafka import KafkaConsumer, KafkaProducer\nproducer = KafkaProducer(bootstrap_servers=[\'broker-1.fqdn:6667\',\n                                            \'broker-2.fqdn:6667\',\n                                            \'broker-3.fqdn:6667\'],\n                        security_protocol="SASL_PLAINTEXT",\n                        sasl_mechanism="GSSAPI",\n                        value_serializer=lambda m: json.dumps(m, cls=HexJsonEncoder).encode(\'utf-8\'))')

get_ipython().run_cell_magic('spark', '', "def getTransactionsInBlock(BLOCKNUM):\n    transactions_in_range=[]\n    transactions_in_block = web3.eth.getBlock(BLOCKNUM,full_transactions=True)['transactions']     \n    for transaction in transactions_in_block:\n        if transaction is not None:\n            cleansesed_transactions=json.dumps(dict(transaction),cls=HexJsonEncoder)     \n            transactions_in_range.append(cleansesed_transactions)\n    return transactions_in_range                \n\ndef produceAllEventLogs(BLOCKNUM,GETH_EVENTS_KAFKA_TOPIC):  \n    for transaction in getTransactionsInBlock(BLOCKNUM):\n        tx_event=dict(web3.eth.getTransactionReceipt(transaction_hash=json.loads(transaction)['hash']))\n        if(tx_event is not None):\n            if(tx_event['logs'] is not None and tx_event['logs']):\n                # Decode every nested tx_log in the tx_event[logs]\n                for tx_log in tx_event['logs']:\n                    tx_json=json.dumps(dict(tx_log), cls=HexJsonEncoder)\n                    producer.send(GETH_EVENTS_KAFKA_TOPIC, tx_json)              ")

get_ipython().run_cell_magic('spark', '', 'kafkatopic="eth_eventlogs"\n\n# Smoke test producing 1 block\'s eventlogs \ncurrentblock = web3.eth.getBlock(\'latest\').number\nproduceAllEventLogs(BLOCKNUM= currentblock,\n                    GETH_EVENTS_KAFKA_TOPIC = kafkatopic )\n\n# Print metrics to verify producer connected successfuly \nproducer.metrics()')

get_ipython().run_cell_magic('spark', '', 'import sys\n\nblockstart= web3.eth.getBlock(\'latest\').number-1\nblockend  = web3.eth.getBlock(\'latest\').number+2\n\nkafkatopic="eth_eventlogs"\n\nprint("Start at block: " + str(blockstart))\ntry:\n    global blockstart\n    while blockstart < blockend:\n        currentblock = web3.eth.getBlock(\'latest\').number\n        if currentblock < blockstart:\n            time.sleep(0.2)\n            pass\n        else:\n            produceAllEventLogs(BLOCKNUM= currentblock,\n                                GETH_EVENTS_KAFKA_TOPIC = kafkatopic ) \n            blockstart=blockstart+1\n            time.sleep(0.2)               \nexcept:\n    print("Unexpected error:", sys.exc_info()[0])\n    pass\nprint("Finished producing block :" + str(blockend))')

get_ipython().run_line_magic('spark', 'delete -s ethlogproducer')

get_ipython().run_line_magic('spark', 'cleanup')



