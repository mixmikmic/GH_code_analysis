import csv
import re
import math
from IPython.display import HTML, display

def cosine_sim(a, b):
    if a == None or b == None:
        return 0
    
    d = dot_product(a, b);
    na = max(1e-20, norm(a));
    nb = max(1e-20, norm(b));
    return (d / (na * nb) + 1) / 2;

def dot_product(a, b):
    result = 0;
    for i in range(len(a)):
        result += (a[i] * b[i])
    return result;
    
def norm(a):
    return math.sqrt(dot_product(a, a))
    
def centroid(a):
    dim = len(a[0])
    num = len(a)
    b = []
    for i in range(dim):
        b.append(0)
        for j in range(num):
            b[i] += (a[j][i] / num)
    return b

def get_clean(s):
    return re.sub('[^0-9a-z\t]+', '', s.lower())
    
def get_pre_punc(s):
    c = s[0].lower()
    o = ord(c)
    if (o < ord('a') or o > ord('z')) and (o < ord('0') or o > ord('9')):
        return c
    else:
        return ""

def get_post_punc(s):
    c = s[len(s)-1].lower()
    o = ord(c)
    if (o < ord('a') or o > ord('z')) and (o < ord('0') or o > ord('9')):
        return c
    else:
        return ""

query = "cambridge"

passage = []
passage.append("The city of Cambridge is a university city and the county town of Cambridgeshire, England. It lies in East Anglia, on the River Cam, about 50 miles (80 km) north of London. According to the United Kingdom Census 2011, its population was 123867 (including 24488 students). This makes Cambridge the second largest city in Cambridgeshire after Peterborough, and the 54th largest in the United Kingdom. There is archaeological evidence of settlement in the area during the Bronze Age and Roman times; under Viking rule Cambridge became an important trading centre. The first town charters were granted in the 12th century, although city status was not conferred until 1951.")
passage.append("Oxford is a city in the South East region of England and the county town of Oxfordshire. With a population of 159994 it is the 52nd largest city in the United Kingdom, and one of the fastest growing and most ethnically diverse. Oxford has a broad economic base. Its industries include motor manufacturing, education, publishing and a large number of information technology and science-based businesses, some being academic offshoots. The city is known worldwide as the home of the University of Oxford, the oldest university in the English-speaking world. Buildings in Oxford demonstrate examples of every English architectural period since the arrival of the Saxons, including the mid-18th-century Radcliffe Camera. Oxford is known as the city of dreaming spires, a term coined by poet Matthew Arnold.")
passage.append("The giraffe (Giraffa camelopardalis) is an African even-toed ungulate mammal, the tallest living terrestrial animal and the largest ruminant. Its species name refers to its camel-like shape and its leopard-like colouring. Its chief distinguishing characteristics are its extremely long neck and legs, its horn-like ossicones, and its distinctive coat patterns. It is classified under the family Giraffidae, along with its closest extant relative, the okapi. The nine subspecies are distinguished by their coat patterns. The scattered range of giraffes extends from Chad in the north to South Africa in the south, and from Niger in the west to Somalia in the east. Giraffes usually inhabit savannas, grasslands, and open woodlands.")

passage_name = []
passage_name.append("Passage about the city of Cambridge")
passage_name.append("Passage about the city of Oxford")
passage_name.append("Passage about giraffes")

passage.append(re.sub(r"\b[Gg]iraffe\b", "Cambridge", passage[2]))
passage.append(re.sub(r"\bCambridge\b", "giraffe", passage[0]))

passage_name.append("Passage about giraffes, but 'giraffe' is replayed by 'Cambridge'")
passage_name.append("Passage about the city of Cambridge, but 'Cambridge' is replaced by 'giraffe'")

doc_all = " ".join(passage)
words = re.sub('[^0-9a-z\t]+', ' ', doc_all.lower()).split()
seen = set()
vocab = [x for x in words if not (x in seen or seen.add(x))]

in_vec = {}
out_vec = {}

with open("data\\in.txt", mode='r') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if row[0] in vocab:
            in_vec[row[0]] = [float(row[i]) for i in range(1, 201)]

with open("data\\out.txt", mode='r') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if row[0] in vocab:
            out_vec[row[0]] = [float(row[i]) for i in range(1, 201)]

def get_font(x):
    f = 13
    x = math.exp(f*(x + 1) / 2) / math.exp(f)
    size = x * 4000
    return "style=\"font-size:{}%;color:#444444\"".format(size)

q_vec = centroid([in_vec[word] for word in query.split()])
display(HTML("<br /><h3>Query: \"cambridge\"</h3>" + "".join("<br /><h4>{}</h4><p style=\"border-left: 6px solid blue; background-color: #fafafa\"><br /><br />".format(passage_name[i]) + " ".join("{}<span {}>{}</span>{}".format(get_pre_punc(word), get_font(cosine_sim(q_vec, out_vec.get(get_clean(word), None))), get_clean(word), get_post_punc(word)) for word in passage[i].split()) + "<br /><br /></p>" for i in range(5))))

