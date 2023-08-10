import dill
from collections import Counter

f=open('clean_haiku_db.txt','r')
s=f.read()

poems = s.split('\n\n\n')
poems = sorted(poems,key= lambda x:len(x.split(' ')))
poems

poems=[p for p in poems if len(p.split(' '))>2 and len(p.split('\n'))==3]
poems

num_poems=len(poems)
num_poems

lengths=[len(p.replace('\n',' ').split(' ')) for p in poems]
sum(lengths)/len(lengths)

len([p for p in poems if len(p.replace('\n',' ').split(' ')) > 23])

t=''.join(p+" " for p in poems)

t=t.lower().replace('\n',' ')
t=''.join(ch for ch in t if ch.isalpha() or ch==' ')
t

tmp=set([w for w in t.split(' ') if len(w)<3])
tmp

themes = Counter([w for w in t.split(' ') if len(w)>2])

themes.most_common(25)

del(themes['the'])
del(themes['with'])
del(themes['and'])
del(themes['from'])
del(themes['for'])
num_themes=len(themes)
num_themes
themes.most_common(10)

separated_poems=[]
for p in poems:
    p_conv=''
    for ch in p:
        if ch.isalpha() or ch==' ':
            p_conv+=ch
        else:
            p_conv+=' '+ch+' '
    p_conv.replace('  ',' ').replace('  ',' ')
    p_conv=' <sop> '+p_conv+' <eop> '
    separated_poems+=[p_conv.lower()]
# what's the average length?
sep_lengths=[len([w for w in p.split(' ') if w != '']) for p in separated_poems]
print("Average length: %.2f" % (sum(sep_lengths)/num_poems,))
print("Max Length: %d" % (max(sep_lengths),))
print(Counter(sorted(sep_lengths)))

short_separated_poems=[p for p in separated_poems if len([w for w in p.split(' ') if w!=''])<33]

tt=(''.join(p+" " for p in short_separated_poems)).replace('  ',' ').replace('  ',' ')
words=Counter([w for w in tt.split(' ') if w != ''])

vocab_size=len(words)
vocab_size

words.most_common(20)

l = [w for w in words if words[w]<5]
len(l)

#for i in range(len(short_separated_poems)):
#    p = short_separated_poems[i]
#    for j in range(len(p)):
#        if p[j] in l:
#            p[j]='<unk>'
for w in l:
    short_separated_poems=[p.replace(' '+w+' ',' <unk> ') for p in short_separated_poems]

tt=(''.join(p+" " for p in short_separated_poems)).replace('  ',' ').replace('  ',' ')
words=Counter([w for w in tt.split(' ') if w != ''])
vocab_size=len(words)
vocab_size

word_to_id={word:id for id,word in enumerate(words)}
id_to_word={id:word for id,word in enumerate(words)}
word_to_id

themes['<unk>']=10000
for thm in list(themes.keys()):
    if thm not in word_to_id:
        print(thm)
        del(themes[thm])
len(themes)

themes['<unk>']

numerical_poems=[[word_to_id[w] for w in p.split(' ') if w != ''] for p in short_separated_poems]
# what's the average length?
sep_lengths=[len(p) for p in numerical_poems]
print("Average length: %.2f" % (sum(sep_lengths)/num_poems,))
print("Max Length: %d" % (max(sep_lengths),))
print(Counter(sorted(sep_lengths)))

theme_list=[list(set([w for w in p if id_to_word[w] in themes])) for p in numerical_poems]
theme_list[:5]
for tl in theme_list:
    if len(tl)==0:
        num_no_th

theme_strengths=[]
for l in theme_list:
    un_norm_strengths=[themes[id_to_word[x]] for x in l]
    total=sum(un_norm_strengths)
    if total > 0:
        normalizer=1./sum(un_norm_strengths)
        strengths = [s*normalizer for s in un_norm_strengths]
    else:
        strengths=un_norm_strengths
    theme_strengths+=[strengths]
theme_strengths[:5]

padded_poems=[p+[word_to_id['<eop>']]*(33-len(p)) for p in numerical_poems]

len(padded_poems[2])

prepared_poems=list(zip(theme_list,theme_strengths,padded_poems))
prepared_poems[1]

id_to_word[1939]

f_out=open('prepared_poem_data.pkl','wb')
dill.dump({'prepared_poems':prepared_poems,'word_to_id':word_to_id,'id_to_word':id_to_word,'themes':themes},f_out)



