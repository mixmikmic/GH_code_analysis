product_costs = {'A':10,'B':20,'C':30,'Others':30}
bill_amount = 0
#Below is a function defintion 
def apply_discount(amount,discount):
    #amount and discount are he parameters to the function
    final_amount = amount - (amount*discount)/100
    #final_amount is the return value from function, if there is nothing you can say return None
    return final_amount

for i in range(5):
    #To take input from users
    product_name  = input("Enter the product Name, A, B, C, NM(No more): ")
    if(product_name == "NM"):
        break
    elif(product_name not in product_costs.keys() ):
        print("Invalid product Name, check it again - ",product_name)
        continue
    elif(product_name in product_costs.keys()):
        no_of_prods = input("Enter the number of products: ")
        bill_amount += int(no_of_prods)*product_costs[product_name]
    

print("\nYour bill before discount - ",bill_amount)
#This is how the function call with arguments or parameters will work  
final_bill_amount = apply_discount(bill_amount,20)
print("Your Final bill after discount - ",final_bill_amount)
print("you saved - ",bill_amount - final_bill_amount)

#No of key value pairs in dictionary. 
print(len(product_costs))

#Converting a squence of keys into a dictonary
col_names = ['Name','Age','Gender','Class']
print(dict.fromkeys(col_names))
print(dict.fromkeys(col_names, "Sample"))

#Power of dictonary lies in providing access through keys
print("Value of a key A - ",product_costs.get('A'))
print("Value of a Key D - ",product_costs.get('D','Not Available'))
#most of the times given a dictonary from other program, we need to check if the key exists before accessing it. 
print('D' in product_costs)
print('B' in product_costs)

#list of items in Dictonaries in the form of tuples, convert to loist to access it 
print(list(product_costs.items()))
#list of keys in dictonary
print(product_costs.keys())

#list of Values in dictionary
print(product_costs.values())#Sometimes when we add key value pairs we are not sure, if there are already a key value in the dictionary
product_costs.setdefault("D",15)
print(product_costs)

