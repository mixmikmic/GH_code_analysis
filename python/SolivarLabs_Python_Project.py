

#Write your project code here
lab_system_details = {'MCA Lab':{'1':['Free','Good'],'2':['Allocated','Repair'],'3':['Free','Good'] }, 'Cisco Lab':{'1':['Free','Good'],'2':['Free','Good'],'3':['Allocated','Repair']}}

def get_list_of_free_systems(lab_name):
    global lab_system_details
    free_system = []
    if lab_name in lab_system_details.keys():
        lab_systems = lab_system_details[lab_name]
        for system_id in lab_systems.keys():
            if lab_systems[system_id][0] == 'Free':
                free_system.append(system_id)

    return free_system

def get_list_of_good_systems(lab_name):
    global lab_system_details
    good_system = []
    if lab_name in lab_system_details.keys():
        lab_systems = lab_system_details[lab_name]
        for system_id in lab_systems.keys():
            if lab_systems[system_id][1] == 'Good':
                good_system.append(system_id)

    return good_system



print('MCA - ',get_list_of_free_systems('MCA Lab'))
print('Cisco - ',get_list_of_good_systems('Cisco Lab'))



