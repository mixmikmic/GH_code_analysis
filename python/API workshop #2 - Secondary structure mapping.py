import re
import requests

base_url = "https://www.ebi.ac.uk/pdbe/"

api_base = base_url + "api/"

secondary_structure_url = api_base + 'pdb/entry/secondary_structure/'

def make_request(url, mode, pdb_id):
    """
    This function can make GET and POST requests to
    the PDBe API
    
    :param url: String,
    :param mode: String,
    :param pdb_id: String
    :return: JSON or None
    """
    if mode == "get":
        response = requests.get(url=url+pdb_id)
    elif mode == "post":
        response = requests.post(url, data=pdb_id)

    if response.status_code == 200:
        return response.json()
    else:
        print("[No data retrieved - %s] %s" % (response.status_code, response.text))
    
    return None

def get_secondary_structure_ranges(pdb_id=None, pdb_list=None):
    """
    This function calls the PDBe API and retrieves the residue
    ranges of secondary structural elements in a single PDB entry
    or in a list of PDB entries
    
    :param pdb_id: String,
    :param pdb_list: String
    :return: None
    """
    # If neither a single PDB id, nor a list was provided,
    # exit the function
    if not pdb_id and not pdb_list:
        print("Either provide one PDB id, or a list of ids")
        return None
    
    if pdb_id:
        # If a single PDB id was provided, call the API with GET
        data = make_request(secondary_structure_url, "get", pdb_id)
    else:
        # If multiple PDB ids were provided, call the API with POST
        # The POST API call expects PDB ids as a comma-separated lise
        pdb_list_string = ", ".join(pdb_list)
        data = make_request(secondary_structure_url, "post", pdb_list_string)
        
    # When no data is returned by the API, exit the function
    if not data:
        print("No data available")
        return None
    
    # Loop through all the PDB entries in the retrieved data
    for entry_id in data.keys():
        entry = data[entry_id]
        molecules = entry["molecules"]
        
        # Loop through all the molecules of a given PDB entry
        for i in range(len(molecules)):
            chains = molecules[i]["chains"]
            
            # Loop through all the chains of a given molecules
            for j in range(len(chains)):
                secondary_structure = chains[j]["secondary_structure"]
                helices = secondary_structure["helices"]
                strands = secondary_structure["strands"]
                helix_list = []
                strand_list = []
                
                # Loop through all the helices of a given chain
                for k in range(len(helices)):
                    start = helices[k]["start"]["residue_number"]
                    end = helices[k]["end"]["residue_number"]
                    helix_list.append("%s-%s" % (start, end))
                
                # Loop through all the strands of a given chain
                for l in range(len(strands)):
                    start = strands[l]["start"]["residue_number"]
                    end = strands[l]["end"]["residue_number"]
                    strand_list.append("%s-%s" % (start, end))
                    
                report = "%s chain %s has " % (entry_id, chains[j]["chain_id"])
                if len(helix_list) > 0:
                    report += "helices at residue ranges %s " % str(helix_list)
                else:
                    report += "no helices "
                report += "and "
                if len(strand_list) > 0:
                    report += "strands at %s" % str(strand_list)
                else:
                    "no strands"
                print(report)
                
    return None

print("Example of a single PDB entry")
get_secondary_structure_ranges(pdb_id="1cbs")
print()
print("Example of multiple PDB entries")
get_secondary_structure_ranges(pdb_list=["2aqa", "2klm"])

