Widget()

from ga4gh.client import protocol
from ga4gh.client import client
c = client.HttpClient("http://1kgenomes.ga4gh.org")

dataset = c.search_datasets().next()
reference_set = c.search_reference_sets().next()
references = [r for r in c.search_references(reference_set_id= reference_set.id)]

contig ={}
for i in references:
    contig[i.name] = str(i.id)

def get_reads_for_name(Name):
    Name = str(Name)
    if type(get_read_groups_by_read_group_set_name(Name)) == str:
        return get_read_groups_by_read_group_set_name(Name)
    else:
        return [i for i in get_read_groups_by_read_group_set_name(Name)]
    
def read_group_set_by_name(name):
    result = None
    for rgs in c.search_read_group_sets(name=name, dataset_id= dataset.id):
        return rgs
    return result
## [name=name, dataset_id= dataset.id]
def get_read_groups_by_read_group_set_name(read_group_set_name):
    if None == read_group_set_by_name(read_group_set_name):
        return "Sorry, bad request for {}".format(read_group_set_name)
    else:
        return read_group_set_by_name(read_group_set_name).read_groups

def chrfunct(chromo):
    chr1 = filter(lambda x: x.name == str(chromo), references)[0]
    return chr1

def cigar_interpreter(sequence, observe, ReferBase):
#     print "Sequence Val: {}".format(sequence)
#     print "Observe Val: {}".format(observe)
#     print "RefereBase Val: {}".format(ReferBase)
    Temp = 0
    BaseCounter = 0
    Variant = ""
    AligSeq = sequence.aligned_sequence
    InterpArr = list([])
    Iter = 0
    type(sequence) 
    for i in sequence.alignment.cigar:
        Length = i.operation_length
        if protocol.CigarUnit.Operation.Name(i.operation) == "ALIGNMENT_MATCH":
            InterpArr[len(InterpArr):len(InterpArr)+Length] = AligSeq[Temp:Temp+Length]
            Temp += Length 
            BaseCounter += Length
            
        elif protocol.CigarUnit.Operation.Name(i.operation) == "CLIP_SOFT":
            Temp += Length
            
     
        elif protocol.CigarUnit.Operation.Name(i.operation) == "DELETE":
            int_iter = 0
            for i in range(Length):
                InterpArr[len(InterpArr) : len(InterpArr)+1] = "N"
                BaseCounter += 1
                int_iter += 1
                if BaseCounter == observe:
                    Variant = ReferBase[BaseCounter:BaseCounter+int_iter]
                    return Variant
                
        elif protocol.CigarUnit.Operation.Name(i.operation) == "INSERT":
            for i in range(Length):
                InterpArr[len(InterpArr):len(InterpArr)+1] = AligSeq[Temp : Temp+1]
                Temp += 1
                if (Temp == observe) and (len(InterpArr) >= Temp+Length+1):
                    Variant = "".join(InterpArr[Temp:Temp+Length+1])
                    return Variant
            
        Iter += 1
    if (Temp >= observe) and (len(sequence.alignment.cigar) == Iter) :
            return InterpArr[observe]
    else: 
        return "N"    
    
    
  

list_of_callset_ids =[]
def find_variants(Start, End, RdGrpSetName, ChromoSm):
    for variant_set in c.search_variant_sets(datasetId=dataset.id):
        if variant_set.name == "phase3-release":
            release = variant_set
            print variant_set
    
    for callSet in c.search_call_sets(variant_set_id= release.id, name= str(RdGrpSetName)):
        mycallset = callSet
        list_of_callset_ids.append(callSet.id)
 
    for variant in c.search_variants(release.id, reference_name=ChromoSm, start=Start, end=End, call_set_ids=list_of_callset_ids):
        print variant
        if len(variant.alternate_bases[0]) == 1 and len(variant.reference_bases) == 1:
            print "\nA VARIANT WAS FOUND"
            print "Variant Name: {}, Start: {}, End: {} \nAlternate Bases: {} \nGenotypes: {}".format(str(variant.names[0]), str(variant.start), str(variant.end), str(variant.alternate_bases[0]), str(variant.calls[0].genotype))
            return 
    return False

def pileUp(contig, position, rgset, Chromosm):
    alleles = []
    rgset = get_reads_for_name(rgset)
    if type(rgset) != str:
        for i in rgset:
            for sequence in c.search_reads(read_group_ids=[i.id],start = position, end = position+1, reference_id=contig):
                if sequence.alignment != None:
                    start = sequence.alignment.position.position
                    observe = position - sequence.alignment.position.position
                    end = start+len(sequence.aligned_sequence)
                    
                    if observe > 100 or observe < 0:
                        continue
                    
                    if len(sequence.alignment.cigar) > 1:
                        allele = cigar_interpreter(sequence, observe,c.list_reference_bases(chrfunct(Chromosm).id, start=start, end= end))      
                    else:
                        allele = sequence.aligned_sequence[observe]
                        
                    alleles.append({"allele": str(allele), "readGroupId":i.id})
        return Calc_Freq(alleles)
    
    else:
        return rgset

def Calc_Freq(Test):
    tot = len(Test)
    AutCalc = {}
    Arr = []
    for i in range(tot):
        if AutCalc.has_key(Test[i]["allele"]) == False and (Test[i]['allele'] != "N"):
            AutCalc.setdefault(Test[i]["allele"], 1)
            Arr.append(Test[i]['allele'])
        else:
            if Test[i]['allele'] == "N":
                tot -= 1
            else:
                AutCalc[Test[i]["allele"]] = float(AutCalc.get(Test[i]["allele"]) + 1)
    Freq = {}
    print "\n{} Reads where used, to determine pile-up".format(tot) 
    tot = float(tot)
    for i in Arr:
        Freq.setdefault(i,float(AutCalc.get(i)/tot))
    return Freq

def Variant_Comp(Position, ReadGroupSetName, Chromosm):
    RdGrp = get_reads_for_name(ReadGroupSetName)
    Chrm = contig.get(Chromosm, None) 
    if (Chrm != None) and type(RdGrp) != (str) :
        base = c.list_reference_bases(Chrm, start = Position, end = Position+1)
        var = pileUp(Chrm, Position, ReadGroupSetName, Chromosm)
        return (str(base), var)
    else:
        if RdGrp == None:
            print"Read Group Set '{}' is not in the API".format(ReadGroupSetName)
        else:
            print"Chromosome '{}' is not in the API".format(Chromosm)

def plot_vars(Position, RdGrpName, Chromo):
    get_ipython().magic('matplotlib inline')
    import matplotlib.pyplot as plt
    Refer, Freqs = Variant_Comp(int(Position), str(RdGrpName),str(Chromo))
    labels = Freqs.keys()
    sizes = Freqs.values()
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    Expl= {}
    Legend = []
    print "Reference Bases:", Refer
    for i in labels:
        if Freqs.get(i) != max(sizes):
            find_variants(int(Position), int(Position)+1, str(RdGrpName), str(Chromo))
            Expl.setdefault(i, .15)
            Legend.append("{}: {} %".format(i, str(Freqs.get(i)*100)[:4]))
        elif i == Refer:
            Expl.setdefault(i,0.8)
            Legend.append("{}: {} %".format(i, str(Freqs.get(i)*100)[:4]))
        else:
            Expl.setdefault(i,0.0)
            Legend.append("{}: {} %".format(i, str(Freqs.get(i)*100)[:4]))
    explode = Expl.values()

    plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=0)
    plt.axis('equal')
    plt.legend(['%s' % str(x) for x in (Legend)])
    plt.show()

def Widget():
    from ipywidgets import widgets
    from ipywidgets import interact
    from IPython.display import display
    
    t0 = widgets.Text(value="Position Exaple:  '120394'", disabled=True)
    text0 = widgets.Text()
    t1 = widgets.Text(value="ReadGroupName Example:  'NA19102'", disabled=True)
    text1 = widgets.Text()
    t2 = widgets.Text(value= "ReferenceSets Example:  '1'", disabled=True)
    text2 = widgets.Text()
    display(t0, text0, t1, text1, t2, text2)
    button = widgets.Button(description="Submit")
    exit = widgets.Button(description="Exit")
    display(button, exit)
    
    
    def exitFunct(c):
        import sys
        sys.exit(["Thank you, you have exited the function"]) 
    
    def Submit(sender):
        Pos, RgSetNm, Chrom = text0.value, text1.value, text2.value
        chr1 = chrfunct(Chrom)
        print "NEXT PLOT VARS FUNCTION W/ PARAMS {}, {}, {}".format(Pos, RgSetNm, Chrom)
        plot_vars(Pos, RgSetNm, Chrom)
        

    def button_clicked(b):
        print "Position: {}, ReadGrpSet: {}, Chrom: {}".format(text0.value, text1.value, text2.value)
        Submit(b)    

    button.on_click(button_clicked)
    exit.on_click(exitFunct)

