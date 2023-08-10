import os

dirname = "../../../../Music/Car Listening/test/"

l =[]
for filename in os.listdir("../../../../Music/Car Listening/test/"):
    print(filename)
    l.append(filename)

def digit_extractor(word):
    for letter in word:
        if letter.isdigit():
            i = word.find(letter)
            j=i
            while (not w[i].isalpha()):
                i+=1
                #print(w[i])
            return (j,i-j,True)
            break
        else:
            return (-1,-1,False)

count=0
for w in l:
    idx, length, flag=digit_extractor(w)
    if flag:
        print("For {}, Digit found at {}. Actual name starts at {}!".format(w,idx,length))
        count+=1
print("\n")
print("Total songs found with digits: ",count)

count=0
for w in l:
    idx, length, flag=digit_extractor(w)
    if flag:
        print("{} processed".format(w))
        os.rename(dirname+w, dirname+w[idx+length:])
        count+=1
print("Total songs found with digits and processed: ",count)

