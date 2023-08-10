def apply_discount(product, discount):
    price = int(product['price'] * (1.0 - discount))
    assert 0 <= price <= product['price']
    return price

#build price dictionary 
shoes = {'name': 'Fancy Shoes', 'price': 14900}

#apply 25% discount 
apply_discount(shoes, 0.25)

apply_discount(shoes, 2.0)



