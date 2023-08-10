import requests
from gbdxtools import Interface
creds = {
    "client_id": "CLIENT_ID",
    "client_secret": "CLIENT_SECRET",
    "username": "my@email.com",
    "password": "password"
}

JUNO_KERNEL_TOKEN = "JUNO_KERNEL_TOKEN"

gbdx = Interface(**creds)
gbdx_headers = {"Authorization": "Bearer {}".format(gbdx.gbdx_connection.token.get("access_token")), "Content-Type": "application/json"}

catalog_ids = ["105001000126EF00", "101001000BAFE000"]

order_id = gbdx.ordering.order(catalog_ids)
data_path = gbdx.ordering.status(order_id)[0]['location']

aoptask = gbdx.Task("AOP_Strip_Processor", data=data_path, bands="MS", enable_acomp=True, enable_pansharpen=False)
junotask = gbdx.Task("Timbr-JunoBaseKernel", JUNO_TOKEN=JUNO_KERNEL_TOKEN)
junotask.inputs.data = aoptask.outputs.data.value
workflow = gbdx.Workflow([aoptask, junotask])
w_id = workflow.execute()

workflow.events

query = """
{
    "state": "running"
}
"""
resp = requests.post("https://geobigdata.io/workflows/v1/workflows/search", headers=gbdx_headers, data=query)
workflows = resp.json()
workflows

gbdx.workflow.cancel(workflows["Workflows"][0])

