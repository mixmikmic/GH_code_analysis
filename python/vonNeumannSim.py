## This is where we instatiate a list. In this case our list will act as MEMORY in or Von Neumann Achitecture
## For a four-bit machine there are only 16 possible addresses
memory = ["cat", "dog", "horse", "mouse", "", "", "", "", "", "", "", "", "", "", "", ""]

## We as the user's will act as the instructions

## We then need to get the next instruction (either READ or WRITE) and the ADDRESS in the memory we are working from

userInput = input("Enter 'READ' to read or 'WRITE' to write \n")
addressInput = int(input("From which address [0-15] \n"))    

## We then need to follow a set of instructions, based on the user's choice, therefore we need an IF statement

if userInput == "READ":
    ## Now we will print what is happening in the Von Neumann model
    print ("CPU sends ", addressInput, " along the ADDRESS bus \n CPU sets CONTROL bus to ", userInput,
           "\n Memory sends '",memory[addressInput], "' along the DATA bus")

elif userInput == "WRITE":
    ## This time we will need to ask the user what DATA they would like to WRITE to the given address, and store this
    writeInput = input("What would you like to store at this location?")
    
    ## Set the data value at ADDRESS in memory to the user's input
    memory[addressInput] = writeInput
    
    ## Now we will print what is happening in the Von Neumann model
    print ("CPU sends ", addressInput, "along the ADDRESS bus. \n CPU sets CONTROL bus to ", userInput,
           "\n CPU sends '", writeInput, "' along the DATA bus")
    
    ## Just to check if it has worked, we will print out the contents of the ADDRESS location
    print ("This is the contents of ADDRESS ", addressInput, "\n", memory[addressInput])
    

## The same program using BINARY

memory = ["cat", "dog", "horse", "mouse", "", "", "", "", "", "", "", "", "", "", "", ""]

userInput = input("Enter 'READ' to read or 'WRITE' to write \n")
addressInput = int(input("From which address [0-15] \n"))   

if userInput == "READ":
    print ("CPU sends ", bin(addressInput), " along the ADDRESS bus \n CPU sets CONTROL bus to ", userInput,
           "\n Memory sends '",memory[addressInput], "' along the DATA bus")

elif userInput == "WRITE":
    writeInput = input("What would you like to store at this location?")
    
    memory[addressInput] = writeInput
    
    print ("CPU sends ", bin(addressInput), "along the ADDRESS bus. \n CPU sets CONTROL bus to ", userInput,
           "\n CPU sends '", writeInput, "' along the DATA bus")
    
    print ("This is the content of ADDRESS ", bin(addressInput), "\n", memory[addressInput])
    



