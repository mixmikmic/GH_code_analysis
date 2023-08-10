import re

# Input parameters
blast_exe        = '/home/chembl/blast/ncbi-blast-2.2.29+/bin/blastp'
query_file       = '/tmp/test.fa'
eval_threshold   = 0.001
num_descriptions = 5 
num_alignments   = 5
database         = '/home/chembl/blast/chembl/chembl_21.fa'

# Output parameters
results_txt      = '/tmp/test.out'
results_xml      = '/tmp/test.xml'
results_csv      = '/tmp/test.csv'


# Query sequence used throughout this tutorial
# ** Feel free to edit the protein sequence below **
# ** DO NOT INCLUDE WHITESPACES IN SEQUENCE HEADER LINE ** 
# **
query_sequence = '''
>Q96P68_OXGR1_HUMAN
MNEPLDYLANASDFPDYAAAFGNCTDENIPLKMHYLPVIYGIIFLVGFPGNAVVISTYIF
KMRPWKSSTIIMLNLACTDLLYLTSLPFLIHYYASGENWIFGDFMCKFIRFSFHFNLYSS
ILFLTCFSIFRYCVIIHPMSCFSIHKTRCAVVACAVVWIISLVAVIPMTFLITSTNRTNR
SACLDLTSSDELNTIKWYNLILTATTFCLPLVIVTLCYTTIIHTLTHGLQTDSCLKQKAR
RLTILLLLAFYVCFLPFHILRVIRIESRLLSISCSIENQIHEAYIVSRPLAALNTFGNLL
LYVVVSDNFQQAVCSTVRCKVSGNLEQAKKISYSNNP
>Q86XF0_DHFRL1_HUMAN
MFLLLNCIVAVSQNMGIGKNGDLPRPPLRNEFRYFQRMTTTSSVEGKQNLVIMGRKTWFS
IPEKNRPLKDRINLVLSRELKEPPQGAHFLARSLDDALKLTERPELANKVDMIWIVGGSS
VYKEAMNHLGHLKLFVTRIMQDFESDTFFSEIDLEKYKLLPEYPGVLSDVQEGKHIKYKF
EVCEKDD
>Q9UKX5_ITGA11_HUMAN
MDLPRGLVVAWALSLWPGFTDTFNMDTRKPRVIPGSRTAFFGYTVQQHDISGNKWLVVGA
PLETNGYQKTGDVYKCPVIHGNCTKLNLGRVTLSNVSERKDNMRLGLSLATNPKDNSFLA
CSPLWSHECGSSYYTTGMCSRVNSNFRFSKTVAPALQRCQTYMDIVIVLDGSNSIYPWVE
VQHFLINILKKFYIGPGQIQVGVVQYGEDVVHEFHLNDYRSVKDVVEAASHIEQRGGTET
RTAFGIEFARSEAFQKGGRKGAKKVMIVITDGESHDSPDLEKVIQQSERDNVTRYAVAVL
GYYNRRGINPETFLNEIKYIASDPDDKHFFNVTDEAALKDIVDALGDRIFSLEGTNKNET
SFGLEMSQTGFSSHVVEDGVLLGAVGAYDWNGAVLKETSAGKVIPLRESYLKEFPEELKN
HGAYLGYTVTSVVSSRQGRVYVAGAPRFNHTGKVILFTMHNNRSLTIHQAMRGQQIGSYF
GSEITSVDIDGDGVTDVLLVGAPMYFNEGRERGKVYVYELRQNLFVYNGTLKDSHSYQNA
RFGSSIASVRDLNQDSYNDVVVGAPLEDNHAGAIYIFHGFRGSILKTPKQRITASELATG
LQYFGCSIHGQLDLNEDGLIDLAVGALGNAVILWSRPVVQINASLHFEPSKINIFHRDCK
RSGRDATCLAAFLCFTPIFLAPHFQTTTVGIRYNATMDERRYTPRAHLDEGGDRFTNRAV
LLSSGQELCERINFHVLDTADYVKPVTFSVEYSLEDPDHGPMLDDGWPTTLRVSVPFWNG
CNEDEHCVPDLVLDARSDLPTAMEYCQRVLRKPAQDCSAYTLSFDTTVFIIESTRQRVAV
EATLENRGENAYSTVLNISQSANLQFASLIQKEDSDGSIECVNEERRLQKQVCNVSYPFF
RAKAKVAFRLDFEFSKSIFLHHLEIELAAGSDSNERDSTKEDNVAPLRFHLKYEADVLFT
RSSSLSHYEVKPNSSLERYDGIGPPFSCIFRIQNLGLFPIHGMMMKITIPIATRSGNRLL
KLRDFLTDEANTSCNIWGNSTEYRPTPVEEDLRRAPQLNHSNSDVVSINCNIRLVPNQEI
NFHLLGNLWLRSLKALKYKSMKIMVNAALQRQFHSPFIFREEDPSRQIVFEISKQEDWQV
PIWIIVGSTLGGLLLLALLVLALWKLGFFRSARRRREPGLDPTPKVLE
>P06804_TNFA_MOUSE
MSTESMIRDVELAEEALPQKMGGFQNSRRCLCLSLFSFLLVAGATTLFCLLNFGVIGPQR
DEKFPNGLPLISSMAQTLTLRSSSQNSSDKPVAHVVANHQVEEQLEWLSQRANALLANGM
DLKDNQLVVPADGLYLVYSQVLFKGQGCPDYVLLTHTVSRFAISYQEKVNLLSAVKSPCP
KDTPEGAELKPWYEPIYLGGVFQLEKGDQLSAEVNLPKYLDFAESGQVYFGVIAL
>P48050_KCNJ4_HUMAN
MHGHSRNGQAHVPRRKRRNRFVKKNGQCNVYFANLSNKSQRYMADIFTTCVDTRWRYMLM
IFSAAFLVSWLFFGLLFWCIAFFHGDLEASPGVPAAGGPAAGGGGAAPVAPKPCIMHVNG
FLGAFLFSVETQTTIGYGFRCVTEECPLAVIAVVVQSIVGCVIDSFMIGTIMAKMARPKK
RAQTLLFSHHAVISVRDGKLCLMWRVGNLRKSHIVEAHVRAQLIKPYMTQEGEYLPLDQR
DLNVGYDIGLDRIFLVSPIIIVHEIDEDSPLYGMGKEELESEDFEIVVILEGMVEATAMT
TQARSSYLASEILWGHRFEPVVFEEKSHYKVDYSRFHKTYEVAGTPCCSARELQESKITV
LPAPPPPPSAFCYENELALMSQEEEEMEEEAAAAAAVAAGLGLEAGSKEEAGIIRMLEFG
SHLDLERMQASLPLDNISYRRESAI
>Q80Z70_SE1L1_RAT
MQVRVRLLLLLCAVLLGSAAASSDEETNQDESLDSKGALPTDGSVKDHTTGKVVLLARDL
LILKDSEVESLLQDEEESSKSQEEVSVTEDISFLDSPNPSSKTYEELKRVRKPVLTAIEG
TAHGEPCHFPFLFLDKEYDECTSDGREDGRLWCATTYDYKTDEKWGFCETEEDAAKRRQM
QEAEAIYQSGMKILNGSTRKNQKREAYRYLQKAAGMNHTKALERVSYALLFGDYLTQNIQ
AAKEMFEKLTEEGSPKGQTGLGFLYASGLGVNSSQAKALVYYTFGALGGNLIAHMVLGYR
YWAGIGVLQSCESALTHYRLVANHVASDISLTGGSVVQRIRLPDEVENPGMNSGMLEEDL
IQYYQFLAEKGDVQAQVGLGQLHLHGGRGVEQNHQRAFDYFNLAANAGNSHAMAFLGKMY
SEGSDIVPQSNETALHYFKKAADMGNPVGQSGLGMAYLYGRGVQVNYDLALKYFQKAAEQ
GWVDGQLQLGSMYYNGIGVKRDYKQALKYFNLASQGGHILAFYNLAQMHASGTGVMRSCH
TAVELFKNVCERGRWSERLMTAYNSYKDDDYNAAVVQYLLLAEQGYEVAQSNAAFILDQR
EATIVGENETYPRALLHWNRAASQGYTVARIKLGDYHFYGFGTDVDYETAFIHYRLASEQ
QHSAQAMFNLGYMHEKGLGIKQDIHLAKRFYDMAAEASPDAQVPVFLALCKLGVVYFLQY
IREANIRDLFTQLDMDQLLGPEWDLYLMTIIALLLGTVIAYRQRQHQDIPVPRPPGPRPA
PPQQEGPPEQQPPQ
>P33277_GAP1_SCHPO
MTKRHSGTLSSSVLPQTNRLSLLRNRESTSVLYTIDLDMESDVEDAFFHLDRELHDLKQQ
ISSQSKQNFVLERDVRYLDSKIALLIQNRMAQEEQHEFAKRLNDNYNAVKGSFPDDRKLQ
LYGALFFLLQSEPAYIASLVRRVKLFNMDALLQIVMFNIYGNQYESREEHLLLSLFQMVL
TTEFEATSDVLSLLRANTPVSRMLTTYTRRGPGQAYLRSILYQCINDVAIHPDLQLDIHP
LSVYRYLVNTGQLSPSEDDNLLTNEEVSEFPAVKNAIQERSAQLLLLTKRFLDAVLNSID
EIPYGIRWVCKLIRNLTNRLFPSISDSTICSLIGGFFFLRFVNPAIISPQTSMLLDSCPS
DNVRKTLATIAKIIQSVANGTSSTKTHLDVSFQPMLKEYEEKVHNLLRKLGNVGDFFEAL
ELDQYIALSKKSLALEMTVNEIYLTHEIILENLDNLYDPDSHVHLILQELGEPCKSVPQE
DNCLVTLPLYNRWDSSIPDLKQNLKVTREDILYVDAKTLFIQLLRLLPSGHPATRVPLDL
PLIADSVSSLKSMSLMKKGIRAIELLDELSTLRLVDKENRYEPLTSEVEKEFIDLDALYE
RIRAERDALQDVHRAICDHNEYLQTQLQIYGSYLNNARSQIKPSHSDSKGFSRGVGVVGI
KPKNIKSSNTVKLSSQQLKKESVLLNCTIPEFNVSNTYFTFSSPSTDNFVIAVYQRGHSK
VLVEVCICLDDVLQRRYASNPVVDLGFLTFEANKLYHLFEQLFLRK
>Q96PD4_IL17F_HUMAN
MTVKTLHGPAMVKYLLLSILGLAFLSEAAARKIPKVGHTFFQKPESCPPVPGGSMKLDIG
IINENQRVSMSRNIESRSTSPWNYTVTWDPNRYPSEVVQAQCRNLGCINAQGKEDISMNS
VPIQQETLVVRRKHQGCSVSFQLEKVLVTVGCTCVTPVIHHVQ
>P10144_GRAB_HUMAN
MQPILLLLAFLLLPRADAGEIIGGHEAKPHSRPYMAYLMIWDQKSLKRCGGFLIRDDFVL
TAAHCWGSSINVTLGAHNIKEQEPTQQFIPVKRPIPHPAYNPKNFSNDIMLLQLERKAKR
TRAVQPLRLPSNKAQVKPGQTCSVAGWGQTAPLGKHSHTLQEVKMTVQEDRKCESDLRHY
YDSTIELCVGDPEIKKTSFKGDSGGPLVCNKVAQGIVSYGRNNGMPPRACTKVSSFVHWI
KKTMKRY
'''

# We will use the query sequence lengths later - just store these for now
query_sequence_details = {}
query_sequence_order   = [];

seq_counter = 0
for seq in query_sequence.split('>'):
    seq = seq.strip(' \n\t')
    if(len(seq) == 0):
        continue
        
    seq_header  = seq.split('\n')[0].strip()
    seq_length  = len(''.join(seq.split('\n')[1:]))
    seq_counter = seq_counter+1
    query_sequence_details[seq_header] = {}
    query_sequence_details[seq_header]['seq_length']  = seq_length
    query_sequence_order.append(seq_header)
    
# Write test query sequence above to query_file location
text_file = open(query_file, "w")
text_file.write(query_sequence)
text_file.close()

# So lets try and run the 'raw' commandline version
get_ipython().system('$blast_exe -query $query_file -db $database -evalue $eval_threshold -num_descriptions $num_descriptions -num_alignments $num_alignments')

# Stdout should be printed below:

from Bio.Blast.Applications import NcbiblastpCommandline

# The outfmt=5 value creates an XML formatted file
blastp_cmd = NcbiblastpCommandline(cmd=blast_exe, query=query_file, db=database, outfmt=5, out=results_xml, evalue=eval_threshold)
stdout, stderr = blastp_cmd()

from Bio.Blast import NCBIXML
result_handle = open(results_xml)
blast_records = NCBIXML.parse(result_handle)

E_VALUE_THRESH = 0.04
result_counter = 0

for blast_record in blast_records:
    for alignment in blast_record.alignments:
         result_counter+=1
         for hsp in alignment.hsps:
             if result_counter <= 5:
                 print '\n# Result ', result_counter, '#'
                 print 'Sequence:   ' + alignment.title
                 print 'Length:    ',  alignment.length
                 print 'E-Value:   ',  hsp.expect
                 print 'Score:     ',  hsp.score
                 print 'Identities:',  hsp.identities
                 print(hsp.query[0:75] + '...')
                 print(hsp.match[0:75] + '...')
                 print(hsp.sbjct[0:75] + '...')

# Create a blast output file in csv format so that it can easily be loaded by pandas
# The outfmt=10 value creates an CSV formatted file
get_ipython().system('$blast_exe -query $query_file -db $database -outfmt 10 -out $results_csv -evalue $eval_threshold')

# Now load BLAST information into pandas dataframe
import pandas
from pandas import DataFrame, read_csv, merge
from pandas.io import sql
from pandas.io.sql import read_sql
# Limit the default pandas dataframe size  
pandas.set_option('display.max_rows', 10)

# Setup database connection to local ChEMBL instance
import psycopg2
con = psycopg2.connect(port=5432, user='chembl', dbname='chembl_21')

Location = results_csv
blast_df = read_csv(Location, names=['query', 'chembl_target_id', 'identity', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'])
blast_df

# Select additional target information from the target_dictionary table 
sql1 = """
select td.chembl_id as chembl_target_id,
       td.pref_name,
       td.organism,
       td.tax_id,
       td.tid,
       td.target_type
 from target_dictionary td
"""

chembl_target_df = read_sql(sql1, con)
chembl_target_df

# We can traverse the ChEMBL activities table to get the count of FDA approved molecules,
# which bind ChEMBL targets with high affinity
sql2 = """
select t.chembl_id as chembl_target_id,
       count(m.chembl_id) as drug_count
  from activities a,
       assays s,
       target_dictionary t,
       molecule_dictionary m
 where a.assay_id=s.assay_id
   and s.tid=t.tid
   and m.molregno=a.molregno
   and a.pchembl_value >= 6
   and s.confidence_score >= 8
   and m.max_phase = 4
   and m.therapeutic_flag=1
group by t.chembl_id
"""

chembl_drug_df = read_sql(sql2, con)
chembl_drug_df

# Similar to the previous query, but this time get the count of 'drug-like' compounds (defined by
# having no rule-of-5 violations), which bind ChEMBL targets with high affinity 
sql3 = """
select t.chembl_id as chembl_target_id,
       count(m.chembl_id) as druglike_count
  from activities a,
       assays s,
       target_dictionary t,
       molecule_dictionary m,
       compound_properties p
 where a.assay_id=s.assay_id
   and s.tid=t.tid
   and m.molregno=a.molregno
   and m.molregno=p.molregno
   and a.pchembl_value >= 6
   and s.confidence_score >= 8
   and p.num_ro5_violations=0
group by t.chembl_id
"""

chembl_druglike_df = read_sql(sql3, con)
chembl_druglike_df

# Get the count of molecules assoicated to a ChEMBL target through a known Mechanism of Action
sql4 = """
select td.chembl_id as chembl_target_id,
       count(distinct dm.molregno) as moa_count
 from drug_mechanism dm, 
      target_dictionary td
where dm.tid=td.tid
group by td.chembl_id
"""

chembl_moa_df = read_sql(sql4, con)
chembl_moa_df

# Carry out the merge and also only return columns we are interested in
rs_merge_df = merge(blast_df, 
                  chembl_target_df,   how='left', on='chembl_target_id' ).merge(
                  chembl_drug_df,     how='left', on='chembl_target_id' ).merge(
                  chembl_druglike_df, how='left', on='chembl_target_id' ).merge(
                  chembl_moa_df,      how='left', on='chembl_target_id')[[
                  'query', 'chembl_target_id','pref_name', 'organism', 'length', 'evalue', 'identity', 'bitscore', 'moa_count', 'drug_count', 'druglike_count' 
                  ]].fillna(0)

rs_merge_df

def druggability_score(query_sequence_length, align_length, identity, moa_count, drug_count, druglike_count):

    align_length = float(align_length)
    identity     = float(identity)   
    
    moa_score      = (align_length/query_sequence_length) * (identity/100) * (1 if (moa_count > 0) else 0)
    drug_score     = (align_length/query_sequence_length) * (identity/100) * (1 if (drug_count > 0) else 0) * 0.8
    druglike_score = (align_length/query_sequence_length) * (identity/100) * (1 if (druglike_count > 0) else 0) * 0.5
    total_score    = round((moa_score + drug_score + druglike_score),2)
    
    return (1 if (total_score > 1) else total_score)

rs_merge_df['druggability_score'] = rs_merge_df.apply(lambda row: druggability_score(query_sequence_details[row['query']]['seq_length'],
                                                                                     row['length'], 
                                                                                     row['identity'], 
                                                                                     row['moa_count'], 
                                                                                     row['drug_count'], 
                                                                                     row['druglike_count']),axis=1)

rs_merge_df

grouped_df = rs_merge_df.groupby('query')['druggability_score'].max().reset_index()
print grouped_df.ix[0]['query']+":",grouped_df.ix[0]['druggability_score']

# Show all results in final table
pandas.set_option('display.max_rows', 500)

druggability_results_df = DataFrame({'query':query_sequence_order}).merge(
                                      grouped_df,
                                      how='left', 
                                      on='query').fillna('No BLAST hits')
druggability_results_df

