get_ipython().system('pip install dicttoxml ')

import json
from dicttoxml import dicttoxml

PETS = {}
PET_STATUS_INDEX = {}
TAG_INDEX = {}
ORDERS = {}
ORDER_STATUS_INDEX = {}
JSON = 'application/json'
XML = 'application/xml'
content_type = JSON

class MissingField(Exception):
    def __init__(self, type_name, field):
        self.msg = '{} is missing required field "{}"'.format(type_name, field)
        
        
class InvalidValue(Exception):
    def __init__(self, name, type_name):
        self.msg = '{} is not a {}'.format(name, type_name)
        
class NotFound(Exception):
    def __init__(self, type_name, id):
        self.msg = 'There is no {} with id {}'.format(type_name, id)


def print_response(content, content_type=JSON):
    if content_type == JSON:
        print(json.dumps(content))
    elif content_type == XML:
        print(dicttoxml(content).decode('UTF-8'))

def split_query_param(param):
    values = []
    for paramValue in param:
        values += paramValue.split(',')
    values = map(lambda x: x.strip(), values)
    return list(values)

def create_error_response(code, error_type, message):
    return {
        'code' : code,
        'type' : error_type,
        'message' : message
    }
        
# Pet APIs
def validate_pet(pet):
    fields = ['id', 'category', 'name', 'photoUrls', 'tags', 'status']
    for field in fields:
        if field not in pet:
            raise MissingField('Pet', field)
            
def persist_pet(pet):
    validate_pet(pet)
    PETS[pet['id']] = pet
    index_pet(pet)
    return pet


def get_pet_by_id(pet_id):
    try:
        pet_id = int(pet_id)
        if not pet_id in PETS:
            raise NotFound('Pet', pet_id)
        else:
            return PETS[pet_id]
    except ValueError:
        raise InvalidValue('Pet id', 'int')

def delete_pet_by_id(pet_id):
    try:
        pet_id = int(pet_id)
        if not pet_id in PETS:
            raise NotFound('Pet', pet_id)
        else:
            pet = PETS[pet_id]
            del PETS[pet_id]
            return pet
    except ValueError:
        raise InvalidValue('Pet id', 'int')

def index_pet(pet):
    # Index the status of the pet
    pet_status = pet['status']
    if pet_status not in PET_STATUS_INDEX:
        PET_STATUS_INDEX[pet_status] = set()
    PET_STATUS_INDEX[pet_status].add(pet['id'])
    
    # index the tags of the pet
    for tag in pet['tags']:
        tag = tag.strip()
        if tag not in STATUS_INDEX:
            TAG_INDEX[tag] = set()
        TAG_INDEX[tag].add(pet['id'])
        
def collect_pets_by_id(petIds):
    petIds = set(petIds)
    petList = []
    for petId in petIds:
        petList.append(PETS[petId])
    return petList

# Order APIs
def validate_order(order):
    fields = ['id', 'petId', 'quantity', 'shipDate', 'status', 'complete']
    for field in fields:
        if field not in order:
            raise MissingField('Order', field)

def persist_order(order):
    validate_order(order)
    ORDERS[order['id']] = order
    
def get_order_by_id(order_id):
    try:
        order_id = int(order_id)
        if not order_id in ORDERS:
            raise NotFound('Order', order_id)
        else:
            return ORDERS[order_id]
    except ValueError:
        raise InvalidValue('Order id', 'int')

def delete_order_by_id(order_id):
    try:
        order_id = int(order_id)
        if not order_id in ORDERS:
            raise NotFound('Order', order_id)
        else:
            order = ORDERS[order_id]
            del ORDERS[order_id]
            return order
    except ValueError:
        raise InvalidValue('Order id', 'int')

REQUEST = json.dumps({
        'body' : {
            'id': 1,
            'category' : { 
                'id' : 1,
                'name' : 'cat'
            },
            'name': 'fluffy',
            'photoUrls': [],
            'tags': ['cat', 'siamese'],
            'status': 'available'
        }
    })

# POST /pet 
try:
    req = json.loads(REQUEST)
    pet = req['body']
    persist_pet(pet)
    response = pet
except MissingField as e:
    response = create_error_response(405, 'Invalid Pet', e.msg)
except ValueError as e:
    response = create_error_response(405, 'Invalid Pet', 'Could not parse json')
finally:
    print_response(response, content_type)

REQUEST = json.dumps({
        'body' : {
            'id': 1,
            'category' : { 
                'id' : 1,
                'name' : 'cat'
            },
            'name': 'fluffy',
            'photoUrls': [],
            'tags': ['cat', 'siamese'],
            'status': 'available'
        }
    })

# PUT /pet 
try:
    req = json.loads(REQUEST)
    new_pet = req['body']
    current_pet = get_pet_by_id(new_pet['id'])
    persist_pet(new_pet)
    response = new_pet
except InvalidValue as e:
    response = create_error_response(400, 'Invalid ID', e.msg)
except ValueError as e:
    response = create_error_response(400, 'Invalid Pet', 'Could not parse json')
except NotFound as e:
    response = create_error_response(404, 'Not Found', e.msg) 
except MissingField as e:
    response = create_error_response(405, 'Invalid Pet', e.msg)
finally:
    print_response(response, content_type)

REQUEST = json.dumps({
        'args' : {
            'status' : ['available , unavailable']
        }
    })

# GET /pet/findByStatus 
req = json.loads(REQUEST)
status_list = split_query_param(req['args']['status'])
pet_ids = []
for status in status_list:
    if status in PET_STATUS_INDEX:
        pet_ids += PET_STATUS_INDEX[status]
pet_list = collect_pets_by_id(pet_ids)
print_response(pet_list, content_type)

REQUEST = json.dumps({
        'args' : {
            'tags' : ['cat , dog, horse']
        }
    })

# GET /pet/findByTags 
req = json.loads(REQUEST)
tag_list = split_query_param(req['args']['tags'])
pet_ids = []
for tag in tag_list:
    if tag in TAG_INDEX:
        pet_ids += TAG_INDEX[tag]
pet_list = collect_pets_by_id(pet_ids)
print_response(pet_list, content_type)

REQUEST = json.dumps({
        'path' : {
            'petId' : 1
        }
    })

# GET /pet/:petId 
try:
    req = json.loads(REQUEST)
    pet_id = req['path']['petId']
    response = get_pet_by_id(pet_id)
except InvalidValue as e:
    response = create_error_response(400, 'Invalid ID', e.msg)
except NotFound as e:
    response = create_error_response(404, 'Not Found', e.msg)    
finally:
    print_response(response, content_type)

REQUEST = json.dumps({
        'path' : {
            'petId' : 1
        },
        'body' : {
            'name' : ['new name']
        }
    })

# POST /pet/:petId 
try:
    req = json.loads(REQUEST)
    pet_updates = req['body']
    pet_id = req['path']['petId']
    old_pet = get_pet_by_id(pet_id)
    props = ['name', 'status']
    for prop in props:
        if prop in pet_updates:
            old_pet[prop] = pet_updates[prop][0]
    response = persist_pet(old_pet)
except InvalidValue as e:
    response = create_error_response(400, 'Invalid ID', e.msg)
except NotFound as e:
    response = create_error_response(404, 'Not Found', e.msg)    
finally:
    print_response(response, content_type)

REQUEST = json.dumps({
        'path' : {
            'petId' : '1'
        }
    })

# DELETE /pet/:petId
try:
    req = json.loads(REQUEST)
    pet_id = req['path']['petId']
    response = delete_pet_by_id(pet_id)
except InvalidValue as e:
    response = create_error_response(400, 'Invalid ID', e.msg)
except NotFound as e:
    response = create_error_response(404, 'Not Found', e.msg)    
finally:
    print_response(response, content_type)

# GET /store/inventory 
status_counts = {}
for status in ORDER_STATUS_INDEX:
    status_counts[status] = len(set(ORDER_STATUS_INDEX[status]))
    
print_response(status_counts, content_type)

REQUEST = json.dumps({
        'body' : {
            'id' : 1,
            'petId' : 1,
            'quantity' : 1,
            'shipDate' : '12/30/2015',
            'status' : 'placed',
            'complete' : False
        }
    })

# POST /store/order 
try:
    req = json.loads(REQUEST)
    order = req['body']
    persist_order(order)
    response = order
except MissingField as e:
    response = create_error_response(400, 'Invalid Order', e.msg)
except ValueError as e:
    response = create_error_response(400, 'Invalid Order', 'Could not parse json')
finally:
    print_response(response, content_type)

REQUEST = json.dumps({
        'path' : {
            'orderId' : 1
        }
    })

# GET /store/order/:orderId
try:
    req = json.loads(REQUEST)
    order_id = req['path']['orderId']
    response = get_order_by_id(order_id)
except InvalidValue as e:
    response = create_error_response(400, 'Invalid ID', e.msg)
except NotFound as e:
    response = create_error_response(404, 'Not Found', e.msg)    
finally:
    print_response(response, content_type)

REQUEST = json.dumps({
        'path' : {
            'orderId' : 1
        }
    })

# DELETE /store/order/:orderId 
try:
    req = json.loads(REQUEST)
    order_id = req['path']['orderId']
    response = delete_order_by_id(order_id)
except InvalidValue as e:
    response = create_error_response(400, 'Invalid ID', e.msg)
except NotFound as e:
    response = create_error_response(404, 'Not Found', e.msg)    
finally:
    print_response(response, content_type)

PETS = {}
STATUS_INDEX = {}
TAG_INDEX = {}
ORDERS = {}

