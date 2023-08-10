def question1(s, t):

    anagram_t = t[::-1]
    if anagram_t in s:
        return True
    else:
        return False

# Test
question1('udacity', 'ad')

def question2(a):
    for i in range(len(a)):
        polindrome = a[i]
        
        for j in len(a):
            if j > i:
                temp = a[i][j]
                polindrome = temp[::-1]
                    

# Test

question2('banana')



