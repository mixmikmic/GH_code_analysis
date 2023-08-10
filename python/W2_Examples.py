s = "Lorem ipsum dolor sit sit amet, consectetur adipiscing elit,      sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.      Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris      nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in      reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla      pariatur. Excepteur sint occaecat cupidatat non proident, sunt in      culpa qui officia deserunt mollit anim id est laborum."
#Parse string into list by every space 
s_parsed=s.split(" ")
print("Total number of words: " + str(len(s_parsed)))
print(s_parsed)
s_parsed.count("sit")

#Define a function that counts how many times a word appears, 
#given a list of unique words.
def num_times_appear(x, entire_list): 
    count=0
    for word in entire_list:
        if word==x: 
            count = count+1 
    return(count)

set_strings = set(s_parsed)
print(set_strings)

dict_words ={x: num_times_appear(x, s_parsed) for x in set_strings}

print("Total number of unique words: "+str(len(dict_words)))
print(dict_words)

dict_words2={x:0 for x in set_strings} #Initialize all to be zero count
for i in s_parsed: 
    dict_words2[i] = dict_words2[i]+1

#Sanity check: 
print(dict_words == dict_words2)
#Returns true!

f=open('/Users/MelodyHuang/Desktop/PythonWorkshop/Week 2/AgathaChristie_TheMysteriousAffair.txt', 'r')
print(f)
myst_affair_text=f.read()

text_parsed=myst_affair_text.split(" ")
print('Total number of words: ' + str(len(text_parsed)))

set_text=set(text_parsed)
print('Total number of unique words: ' + str(len(set_text)))

dict_text={x:num_times_appear(x, text_parsed) for x in set_text}

dict_text2={x:0 for x in set_text}
for i in text_parsed: 
    dict_text2[i]=dict_text2[i]+1

print(dict_text2)





