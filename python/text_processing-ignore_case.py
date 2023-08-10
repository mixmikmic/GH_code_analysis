import re
#import enchant
import os
import sys
import glob
import string
import time
import operator
from collections import OrderedDict, Counter
from IPython.display import clear_output
#%pylab inline

pwd()

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
define pastas de trabalho
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
path = 'C:/Users/marcelo.ribeiro/Documents/textfiles-original-regrouped/' #path_input
path2 = 'C:/Users/marcelo.ribeiro/Documents/textfiles-corrected-ignorecase/' #path_output
#path = '../textfiles/' #path_input
#path2 = '../textfiles-corrected' #path_output

onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and not f.startswith('.')]
onlyfiles.sort()

onlyfiles[0:4]

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
data cleansing dos arquivos .txt
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
count = 0
percentil = int(len(onlyfiles)/100)
freqtotal = Counter()
for txt in onlyfiles[0:]:
    with open(os.path.join(path,txt), 'r', encoding='utf-8') as f:
        count += 1
        if count % percentil == 0: print(int(count/percentil),'% done')
        if (count) % (percentil-1) == 0: clear_output()
        texto = f.read()
        
        ######PRINCIPAIS (MAIOR EFEITO)######
        texto = re.sub('—', r'-',texto)
        texto = re.sub('(?<!exm|.ex|..\d)ª\n*',r'a',texto, flags=re.I) # corrige vários, mas arriscado, principalmente com '+' ou '*'
        texto = re.sub('(?<!exm|.ex|..\d)ª',r'a',texto, flags=re.I) # correção diferente; resolve overlapping anterior
        texto = re.sub('nº\n*(\d)',r'nº \1',texto, flags=re.I) # separa nº do número respectivo
        texto = re.sub('(?<!exm|.ex|..\d)º\n*',r'o',texto, flags=re.I) # corrige vários, mas arriscado, principalmente com '+' ou '*' ver xmº
        texto = re.sub('(?<!exm|.ex|..\d)º.\D',r'o',texto, flags=re.I) # correção diferente; resolve overlapping anterior
        texto = re.sub('([a-z])-\n([a-z])',r'\1\2',texto, flags=re.I) # corrige vários, mas arriscado, principalmente com '\n+' ou '\n*'
        
        ######SÍMBOLOS######
        texto = re.sub('([^\W\d])\»\n([^\W\d])',r'\1 \2',texto, flags=re.I) # alternativas: 'i, 'juntar'
        texto = re.sub('([^\W\d])\«\n([^\W\d])',r'\1 \2',texto, flags=re.I) # alternativas: 'i, 'juntar'
        texto = re.sub('([^\W\d])\"([^\W\d])',r'\1 \2',texto, flags=re.I) # alternativas: 'm', 'n', 'juntar' 
        texto = re.sub('[\“\”\"\'\«\»]{2,}',r'"',texto, flags=re.I) # símbolos repetidos
        texto = re.sub('(\“|\”|\«|\»)',r'"',texto, flags=re.I) # conserta aspas
        texto = re.sub('([^\W\d])\«([^\W\d])',r'\1-\2',texto, flags=re.I) # alternativas: 'i, 'espaço'
        texto = re.sub('([^\W\d])\»([^\W\d])',r'\1 \2',texto, flags=re.I) # alternativas: 'i, 'hífen'        

        texto = re.sub('([^\W\d_])\_([^\W\d_])',r'\1 \2',texto, flags=re.I) # alternativas: juntar  
        texto = re.sub('([^\W\d])\!([^\W\d])',r'\1i\2',texto, flags=re.I) # alternativas: 'í'
        texto = re.sub('([^\W\d])\;([áàaâã]o|[óòoôõ]e)',r'\1ç\2',texto, flags=re.I) # alternativas: 'ç', 'ã', 'i', 'í'
        texto = re.sub('([^\W\d])\;(os?\W)',r'\1ã\2',texto, flags=re.I) 
        texto = re.sub('([^\W\d])\;([^\W\d])',r'\1i\2',texto, flags=re.I) 
        texto = re.sub('([^\W\d])\:([^\W\d])',r'\1i\2',texto, flags=re.I) # alternativas: 'í'
        texto = re.sub('([^\W\d])\“([^\W\d])',r'\1n\2',texto, flags=re.I) # alternativas: 'm', 'espaço'
        texto = re.sub('“',r'"',texto, flags=re.I) # a troca de aspas corrige palavras
        texto = re.sub('([^\W\d])\”([^\W\d])',r'\1n\2',texto, flags=re.I) # alternativas: '' ', 'espaço'
        texto = re.sub('”',r'"',texto, flags=re.I) # a troca de aspas corrige palavras
        texto = re.sub('([^\W\d])\(([^\W\d])',r'\1 (\2',texto, flags=re.I) 
        texto = re.sub('([^\W\d])\)([^\W\d])',r'\1) \2',texto, flags=re.I) 
        texto = re.sub('([^\W\d])\$([^\W\d])',r'\1s\2',texto, flags=re.I)
        #outros simbolos: £, %
        
        ######PONTUACAO ANTES DE LETRA######
        texto = re.sub('(\.)([^\s0-9])',r'\1 \2',texto, flags=re.I)
        texto = re.sub('(\,)([^\s0-9])',r'\1 \2',texto, flags=re.I)
        
        ######NUMEROS######
        texto = re.sub('([^\W\d])1([^\W\d])',r'\1i\2',texto, flags=re.I) # alternativas: 'l, 'í'
        texto = re.sub('([^\W\d])1([^\W\d])',r'\1i\2',texto, flags=re.I) #needed to repeat: ver questão do overlapping
        texto = re.sub('([^\W\d])1([^\/ªº][\D])',r'\1l\2',texto) #
        texto = re.sub('([\D][^\/])1([^\W\d])',r'\1l\2',texto) #
        texto = re.sub('([^\W\d])5([^\/ªº][\D])',r'\1s\2',texto, flags=re.I) # alternativas: 'sí'
        texto = re.sub('([\D][^\/])5([^\W\d])',r'\1s\2',texto, flags=re.I) # alternativas: 'sí'
        texto = re.sub('([^\W\d])0([^\/][\D])',r'\1o\2',texto, flags=re.I)
        texto = re.sub('([\D][^\/])0([^\W\d])',r'\1o\2',texto, flags=re.I)
        texto = re.sub('([\D][^\/])11([^\W\d])',r'\1li\2',texto, flags=re.I)

        texto = re.sub('(3)(rd)([^\w][^\d])',r'\1 \2\3',texto, flags=re.I)
        texto = re.sub('(2)(nd)([^\w][^\d])',r'\1 \2\3',texto, flags=re.I)
        texto = re.sub('(1)(st)([^\w][^\d])',r'\1 \2\3',texto, flags=re.I)
        texto = re.sub('([04-9])(th)([^\w][^\d])',r'\1 \2\3',texto, flags=re.I)
        
        ######TERMOS COM 'ANTI' E 'NAO'######
        texto = re.sub('n[áàaâ]oio', r'nacio',texto, flags=re.I)
        texto = re.sub('n[áàaâ]oue', r'naque',texto, flags=re.I) 
        texto = re.sub('n[áàâã]o(pro[l1]|ferro|al[íìiî]|[íìiî]nterv|nuc[l1]|d[íìiî]scr|exp[l1]|reco[nmh]|imp[l1]|gov|indu|redu|rec[íìiî]p|memb|aut|ren|s[íìiî]g|[íìiî]ng|tar|intro|ofi)', r'não-\1',texto, flags=re.I)
        texto = re.sub('anti(nuc|com|ame|trus|castr|col|cl[ií]|sem|eco|arm|per|apar|infl|disc|rac|chi|imp|sio|zio|sub|dum|eur|hig|dem|oci|port|tan|air|gov|pol|ter|sat|sov|ét|per|rus|a[eé]|res|mil|d[eé]t|fas|hit|bra|nav|cic|rec|isr)', r'anti-\1',texto, flags=re.I)        
        
        ######TERMOS QUEBRADOS######
        texto = re.sub('bras\w\we\w[^\w]{0,3}\n+r([ao]s?[^\w])', r'brasileir\1',texto, flags=re.I)
        texto = re.sub('([inaro])[^\w]{0,3}\n+c[íìiîl0-9]al([^\w])', r'\1cial\2',texto, flags=re.I) 
        texto = re.sub('([napi])[^\w]{0,3}\n+c[íìiîl0-9]ona[l1i]([^\w])', r'\1cional\2',texto, flags=re.I) 
        texto = re.sub('e[^\w]{0,3}\n+s[íìiîl0-9]den[íìiîl0-9]e([^\w])', r'esidente\1',texto, flags=re.I)
        texto = re.sub('m\w[^\w]{0,3}\n+n[íìiî]st\w([oa]s?[^\w])', r'ministr\1',texto, flags=re.I)
        texto = re.sub('([ai])[^\w]{0,3}\n+l[íìiî]dade([^\w])', r'\1lidade\2',texto, flags=re.I)
        texto = re.sub('([lora])[^\w]{0,3}\n+v[íìiî]men\wo([^\w])', r'\1vimento\2',texto, flags=re.I)
        texto = re.sub('([e])[^\w]{0,3}\n+senvolv[íìiî]men([^\w])', r'\1senvolvimen\2',texto, flags=re.I)
        texto = re.sub('(a\w)[^\w]{0,3}\n+gen\w[íìiî]n([oa]s?[^\w])', r'argentin\2',texto, flags=re.I)
        texto = re.sub('([^\w])pres[íìiî]den[^\w]{0,3}\n+([ct])', r'\1presiden\2',texto, flags=re.I)
        texto = re.sub('([^\w])desenvo\w[^\w]{0,3}\n+([vt])', r'\1desenvol\2',texto, flags=re.I)
        texto = re.sub('([^\w])gove\w[^\w]{0,3}\n+([aoim])', r'\1gover\2',texto, flags=re.I) 
        texto = re.sub('([a-z])[^a-z]ção', r'\1ção',texto, flags=re.I)
        texto = re.sub('(ç[áàâã]o|ções|ências|éias|érios|ível|ssão|ômic[ao]s|ípios)([^\W\d])',r'\1 \2',texto, flags=re.I) # aumenta erros no total por lidar com erros graves
        texto = re.sub('(ência|éia|ério|ômic[ao]|ípio)([^\W\ds])',r'\1 \2',texto, flags=re.I) # 
        
        ######TERMOS COM ERROS SIMPLES (TROCA DE LETRA, FALTA DE ACENTOS)######
        texto = re.sub('([^a-z])n[áàaâã]o([^a-z])', r'\1não\2',texto, flags=re.I)
        texto = re.sub('([^a-z])pa[íiìî0-9]ses([^a-z])', r'\1países\2',texto, flags=re.I)
        texto = re.sub('rela[cç][óòoôõ\w]es', r'relações',texto, flags=re.I)
        texto = re.sub('brasil([eé][^a-z][^i])', r'brasil \1',texto, flags=re.I)
        texto = re.sub('exce[l1][èéê\w]nc[íiìî0-9]a', r'excelência',texto, flags=re.I)
        texto = re.sub('([^a-z])s[áàaâã]o([^a-z])', r'\1são\2',texto, flags=re.I)
        texto = re.sub('([^a-z])reuni[áàaâã]o([^a-z])', r'\1reunião\2',texto, flags=re.I)
        texto = re.sub('assemb[l1][èéeê][íiìî0-9]a', r'assembleia',texto, flags=re.I)
        texto = re.sub('brasilem', r'brasil em',texto, flags=re.I)
        texto = re.sub('minist[èéeê]r[íiìî0-9]o', r'ministério',texto, flags=re.I)
        texto = re.sub('c\w\wpe\wa[cç][aàââã]o', r'cooperação',texto, flags=re.I)
        texto = re.sub('com\wss[áàaâã]o([^a-z])', r'comissão\1',texto, flags=re.I)
        texto = re.sub('mat[èéeê]r[íiìîl0-9]a([^a-z])', r'matéria\1',texto, flags=re.I)
        texto = re.sub('([^a-z])quest[áàaâã]o([^a-z])', r'\1questão\2',texto, flags=re.I)
        texto = re.sub('([^a-z])rela[cç][áàaâã]o([^a-z])', r'\1relação\2',texto, flags=re.I)
        texto = re.sub('([^a-z])secreto', r'\1 secreto',texto, flags=re.I)
        texto = re.sub('([^a-z])reservado', r'\1 reservado',texto, flags=re.I)
        texto = re.sub('petr[óòoôõ\w]leo', r'petróleo',texto, flags=re.I)
        texto = re.sub('econ[óòoôõ\w]mic([ao]s?[^a-z])', r'econômic\1',texto, flags=re.I)
        texto = re.sub('rep[úùuû]b(u|li)ca', r'república',texto, flags=re.I)
        texto = re.sub('delega[cç][áàaâã]o', r'delegação',texto, flags=re.I)
        texto = re.sub('na[çc][óòoôõ]es', r'nações',texto, flags=re.I)
        texto = re.sub('([^a-z])est[áàaâã]o([^a-z])', r'\1estão\2',texto, flags=re.I)
        texto = re.sub('([^a-z])per[ìíiî]odo([^a-z])', r'\1período\2',texto, flags=re.I)
        texto = re.sub('retransm[íìiîl0-9]ss?[áàaâã]o', r'retransmissão',texto, flags=re.I)
        texto = re.sub('import[áàaâã]ncia', r'importância',texto, flags=re.I)
        texto = re.sub('negoc[íìiîl0-9]a[cç][óòoôõ\w]es', r'negociações',texto, flags=re.I)
        texto = re.sub('([^a-z])m[íìiîl0-9]ss[áàaâã]o([^a-z])', r'\1missão\2',texto, flags=re.I)
        texto = re.sub('urgent[[íìiîl0-9]]ss[íìiîl0-9]mo', r'urgentíssimo',texto, flags=re.I)
        texto = re.sub('bras[ìiîl0-9]l([b-df-hj-z0-9])', r'brasil \1',texto, flags=re.I)
        texto = re.sub('assun[çc][áàaâã]o', r'assunção',texto, flags=re.I)
        texto = re.sub('dec[l1]ara[cç][óòoôõ]es', r'declarações',texto, flags=re.I)
        texto = re.sub('([^a-z])ostens\wvo', r'\1 ostensivo',texto, flags=re.I)
        texto = re.sub('conversa[cç][óòoôõ]es', r'conversações',texto, flags=re.I)
        texto = re.sub('condi[cç][óòoôõ]es', r'condições',texto, flags=re.I)
        texto = re.sub('brasi ([^l])', r'brasil \1',texto, flags=re.I)
        texto = re.sub('pre[saã][íìiîl0-9]?dente', r'presidente',texto, flags=re.I)
        texto = re.sub('pr[íìiîl0-9]nc[íìiîl0-9]p[íìiîl0-9]o(s?)', r'princípio\1',texto, flags=re.I)
        texto = re.sub('[íìiîl0-9]nstru[cç][óòoôõ\w]es', r'instruções',texto, flags=re.I)
        texto = re.sub('terr[íìiîl0-9]t[óòoôõ]rio(s?)', r'território\1',texto, flags=re.I)
        texto = re.sub('seguranca', r'segurança',texto, flags=re.I)
        texto = re.sub('neg[óòoôõ]c[íìiîl0-9]os', r'negócios',texto, flags=re.I)
        texto = re.sub('[íìiîl0-9]nforma[cç][óòoôõ]es', r'informações',texto, flags=re.I)
        texto = re.sub('sess[áàaâã]o', r'sessão',texto, flags=re.I)
        texto = re.sub('([^a-z])ultimos([^a-z])', r'\1últimos\2',texto, flags=re.I)
        texto = re.sub('([^a-z])jap[áàaâã]o([^a-z])', r'\1Japão\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('([^a-z])jo[áàaãâ]o([^a-z])', r'\1João\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('part[íìiîl0-9]c[íìiîl0-9]pa[cça][áàaãâ]o', r'participação',texto, flags=re.I)
        texto = re.sub('([^a-z])op[íìiîl0-9]n[íìiîl0-9][áàaâã]o([^a-z])', r'\1opinião\2',texto, flags=re.I)
        texto = re.sub('([^a-z])v[áàaâã]rios([^a-z])', r'\1vários\2',texto, flags=re.I)
        texto = re.sub('t[éèeê]cn[íìiîl0-9]c([ao]s?[^a-z])', r'técnic\1',texto, flags=re.I)
        texto = re.sub('([^a-z])[áàaâã]guas([^a-z])', r'\1águas\2',texto, flags=re.I)
        texto = re.sub('reso[l1]u[cç][áàaâã]o', r'resolução',texto, flags=re.I)
        texto = re.sub('dec[l1]ara[cç][áàaâã]o', r'declaração',texto, flags=re.I)
        texto = re.sub('[íìiîl0-9]ndepend[éèeê]ncia', r'independência',texto, flags=re.I)
        texto = re.sub('([^a-z])ara[úùuû]jo([^a-z])', r'\1Araújo\2',texto, flags=re.I)
        texto = re.sub('([^a-z])un[íìiîl0-9][áàaâã]o([^a-z])', r'\1união\2',texto, flags=re.I)
        texto = re.sub('[íìiîl0-9]nforma[cç][áàaâã]o', r'informação',texto, flags=re.I)
        texto = re.sub('quest[óòoôõ]es', r'questões',texto, flags=re.I)
        texto = re.sub('([^a-z])regi[áàaâã0-9]o([^a-z])', r'\1região\2',texto, flags=re.I)
        texto = re.sub('portugu[èéê\w]s([^ae])', r'português\1',texto, flags=re.I)
        texto = re.sub('d[íìiîl0-9]f[íìiî]c[íìiîl0-9]l([^a-z])', r'difícil\1',texto, flags=re.I)
        texto = re.sub('([^a-z])precos([^a-z])', r'\1preços\2',texto, flags=re.I)
        texto = re.sub('([^a-z])conf[íìiîl0-9]denc[íìiîl0-9]a[il1]([^a-z])', r'\1 confidencial\2',texto, flags=re.I)
        texto = re.sub('pos[íìiîl0-9][cç][óòôoõ]es', r'posições',texto, flags=re.I)
        texto = re.sub('pres[íìiîl0-9]d[éèeê]ncia([^a-z])', r'presidência\1',texto, flags=re.I)
        texto = re.sub('constru[cç][áàaâã]o', r'construção',texto, flags=re.I)
        texto = re.sub('relat[óòoôõ]rio', r'relatório',texto, flags=re.I)
        texto = re.sub('reuníìiîl0-9[óòoôõ]es', r'reuniões',texto, flags=re.I)
        texto = re.sub('([^a-z])ur[áàaâã\w]nio([^a-z])', r'\1urânio\2',texto, flags=re.I)
        texto = re.sub('preocupa[cç][áàaâã]o', r'preocupação',texto, flags=re.I)
        texto = re.sub('([^a-z])cor[éèeê][íìiîl0-9]a([^a-z])', r'\1Coreia\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('organ[íìiîl0-9]za[cç][áàaâã]o', r'organização',texto, flags=re.I)
        texto = re.sub('sov[íìiîl0-9][éèeê]t[íìiîl0-9]c([ao]s?)', r'soviétic\1',texto, flags=re.I)
        texto = re.sub('necess[áàaâã]r[íìiîl0-9]([ao]s?[^a-z])', r'necessári\1',texto, flags=re.I)
        texto = re.sub('([^a-z])c[áàaâã]mara([^a-z])', r'\1câmara\2',texto, flags=re.I)
        texto = re.sub('func[íìiîl0-9]on[áàaâã]ri([ao]s?)', r'funcionári\1',texto, flags=re.I)
        texto = re.sub('produ[cç][áàaâã]o', r'produção',texto, flags=re.I)
        texto = re.sub('sa[úùuû]de', r'saúde',texto, flags=re.I)
        texto = re.sub('cr[íìiîl0-9]a[cç][áàaâã]o', r'criação',texto, flags=re.I)
        texto = re.sub('aud[íìiîl0-9][éèeê]ncia', r'audiência',texto, flags=re.I)
        texto = re.sub('([^a-z])[íìiîl0-9]t[áàaâã]lia([^a-z])', r'\1Itália\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('or[íìiîl0-9]enta[cç][áàaâã]o', r'orientação',texto, flags=re.I)
        texto = re.sub('([^a-z])rea[cç][áàaâã]o([^a-z])', r'\1reação\2',texto, flags=re.I)
        texto = re.sub('([^a-z])[íìiîl0-9]nd[íìiîl0-9]ce([^a-z])', r'\1índice\2',texto, flags=re.I)
        texto = re.sub('admin[íìiîl0-9]stra[cç][áàaâã]o', r'administração',texto, flags=re.I)
        texto = re.sub('([^\s])azeredo', r'\1 Azeredo',texto, flags=re.I)
        texto = re.sub('exporta[cç][áàaâã]o', r'exportação',texto, flags=re.I)
        texto = re.sub('opera[cç][áàaâã]o', r'operação',texto, flags=re.I)
        texto = re.sub('hip[óòoôõ]tese', r'hipótese',texto, flags=re.I)
        texto = re.sub('real[íìiîl0-9]za[cç][áàaâã]o', r'realização',texto, flags=re.I)
        texto = re.sub('tel[èéeê]g\wama', r'telegrama',texto, flags=re.I)
        texto = re.sub('ele[íìiîl0-9][cç][óòoôõ\w]es', r'eleições',texto, flags=re.I)
        texto = re.sub('gest[áàaâã]o', r'gestão',texto, flags=re.I)
        texto = re.sub('delega[cç][óòoôõ]es', r'delegações',texto, flags=re.I)
        texto = re.sub('europ[èéeê][íìiîl0-9]a', r'europeia',texto, flags=re.I)
        texto = re.sub('transfer[èéeê]ncia', r'transferência',texto, flags=re.I)
        texto = re.sub('montev[íìiîl0-9]d[èéeê]u', r'Montevidéu',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('concess[áàaâã]o', r'concessão',texto, flags=re.I)
        texto = re.sub('exporta[cç][óòoôõ\w]es', r'exportações',texto, flags=re.I)
        texto = re.sub('bras[íìiîl0-9]l—', r'brasil -',texto, flags=re.I)
        texto = re.sub('comun[íìiîl0-9]ca[cç][áàaâã]o', r'comunicação',texto, flags=re.I)
        texto = re.sub('([^a-z])g[óòoôõ0-9]v[èéeê]rno([^a-z])', r'\1governo\2',texto, flags=re.I)
        texto = re.sub('esforcos', r'esforços',texto, flags=re.I)
        texto = re.sub('atua[cç][áàaâã]o', r'atuação',texto, flags=re.I)
        texto = re.sub('negoc[íìiîl0-9]a[cç][áàaâã]o', r'negociação',texto, flags=re.I)
        texto = re.sub('[íìiîl0-9]mpress[áàaâã]o', r'impressão',texto, flags=re.I)
        texto = re.sub('na[cç][áàaâã]o', r'nação',texto, flags=re.I)
        texto = re.sub('alem[áàaâã]o', r'alemão',texto, flags=re.I)
        texto = re.sub('[íìiîl0-9]nterven[cç][áàaâã]o', r'intervenção',texto, flags=re.I)
        texto = re.sub('[íìiîl0-9]nforma[cç].es', r'informações',texto, flags=re.I)
        texto = re.sub('aprova[cç][áàaâã]o', r'aprovação',texto, flags=re.I)
        texto = re.sub('[íìiîl0-9]ndica[cç][áàaâã]o', r'indicação',texto, flags=re.I)
        texto = re.sub('d[èéeê]tente', r'détente',texto, flags=re.I)
        texto = re.sub('opos[íìiîl0-9][cç][áàaâã]o', r'oposição',texto, flags=re.I)
        texto = re.sub('equ[íìiîl0-9]l[íìiî]brio', r'equilíbrio',texto, flags=re.I)
        texto = re.sub('camili[óòoôõ]n', r'Camilión',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('ed[íìiîl0-9][cç][áàaâã]o', r'edição',texto, flags=re.I)
        texto = re.sub('colabora[cç][áàaâã]o', r'colaboração',texto, flags=re.I)
        texto = re.sub('pos[íìiîl0-9][cç][áàaâã]o', r'posição',texto, flags=re.I)
        texto = re.sub('comun[íìiîl0-9]ca[cç][óòoôõ]es', r'comunicações',texto, flags=re.I)
        texto = re.sub('([^a-z])\wuest[áàaâã]o([^a-z])', r'\1questão\2',texto, flags=re.I)
        texto = re.sub('expans[áàaâã]o', r'expansão',texto, flags=re.I)
        texto = re.sub('([^a-z])[óòoôõ]rg[aáàâã]o(s?[^a-z])', r'\1órgão\2',texto, flags=re.I)
        texto = re.sub('explora[cç][áàaâã]o', r'exploração',texto, flags=re.I)
        texto = re.sub('([^a-z])di[áàaâã]rio([^a-z])', r'\1diário\2',texto, flags=re.I)
        texto = re.sub('bras[ìiîl0-9]la([a-z])', r'Brasil a\1',texto, flags=re.I)
        texto = re.sub('urg[èéeê]nc[íìiîl0-9]a', r'urgência',texto, flags=re.I)
        texto = re.sub('just[íìiîl0-9][cç]a', r'justiça',texto, flags=re.I)
        texto = re.sub('serv[íìiîl0-9][cç]o', r'serviço',texto, flags=re.I)
        texto = re.sub('d[íìiîl0-9]v[íìiîl0-9]s[áàaâã]o', r'divisão',texto, flags=re.I)
        texto = re.sub('(\/?)brase[hnm]b', r'\1Brasemb',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('te[l1]egr[áàaâã][^f\W]a', r'telegrama',texto, flags=re.I)
        texto = re.sub('[íìiîl0-9]mporta[cç][áàaâã]o', r'importação',texto, flags=re.I)
        texto = re.sub('serv[íìiîl0-9]cos', r'serviços',texto, flags=re.I)
        texto = re.sub('echeverr[íìî\w]a', r'Echeverría',texto, flags=re.I)
        texto = re.sub('([^a-z])ac[óòoôõ]rdo([^a-z])', r'\1acordo\2',texto, flags=re.I)
        texto = re.sub('c[íìiîl0-9][èéeê]ncia', r'ciência',texto, flags=re.I)
        texto = re.sub('combust[íìî\w]vel', r'combustível',texto, flags=re.I)
        texto = re.sub('([^d\W\d]{3})a[mn][èéeê]r\wca[mn]', r'\1-american',texto, flags=re.I)
        texto = re.sub('inter-american', r'interamerican',texto, flags=re.I) #NOVO 07-28, conserta anterior
        texto = re.sub('lat[íìiîl0-9]no-?american([ao]s?[^a-z])', r'latino-american\1',texto, flags=re.I)
        texto = re.sub('t[óòoôõ]qu[íìiîl0-9]o', r'Tóquio',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('rela[\w]ões', r'relações',texto, flags=re.I)
        texto = re.sub('([^a-z])clar[íìiî]n([^a-z])', r'\1Clarín\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('telev[íìiîl0-9]s[áàaâã]o', r'televisão',texto, flags=re.I)
        texto = re.sub('man[íìiîl0-9]festa[cç][áàaâã]o', r'manifestação',texto, flags=re.I)
        texto = re.sub('[íìiîl0-9]mporta[cç][óòoôõ\w]es', r'importações',texto, flags=re.I)
        texto = re.sub('dafontoura', r'da Fontoura',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('m[íìiîl0-9]ss[óòoôõ]es', r'missões',texto, flags=re.I)
        texto = re.sub('japon[èéeê]s([^a-z])', r'japonês\1',texto, flags=re.I)
        texto = re.sub('exped[íìiîl0-9]doem', r'expedido em',texto, flags=re.I)
        texto = re.sub('d[íìiîl0-9]re[cç][áàaâã]o', r'direção',texto, flags=re.I)
        texto = re.sub('compet[èéeê]nc[íìiîl0-9]a', r'competência',texto, flags=re.I)
        texto = re.sub('[íìiîl0-9]nstala[cç][áàaâã]o', r'instalação',texto, flags=re.I)
        texto = re.sub('jur[íìiîl0-9]d[íìiîl0-9]c([ao]s?[^a-z])', r'jurídic\1',texto, flags=re.I)
        texto = re.sub('restr[íìiîl0-9][cç][óòoôõ]es', r'restrições',texto, flags=re.I)
        texto = re.sub('esforco', r'esforço',texto, flags=re.I)
        texto = re.sub('br[íìiîl0-9]t[áàaâã\w]nic([ao]s?)', r'britânic\1',texto, flags=re.I)
        texto = re.sub('aproxima[cç][áàaâã]o', r'aproximação',texto, flags=re.I)
        texto = re.sub('plen[áàaâã]rio', r'plenário',texto, flags=re.I)
        texto = re.sub('const[íìiîl0-9]tu[íìiîl0-9][cç][áàaâã]o', r'constituição',texto, flags=re.I)
        texto = re.sub('manuten[cç][áàaâã]o', r'manutenção',texto, flags=re.I)
        texto = re.sub('c[íìiî]rculos', r'círculos',texto, flags=re.I)
        texto = re.sub('d[íìiîl0-9]vulga[cç][áàaâã]o', r'divulgação',texto, flags=re.I)
        texto = re.sub('exter[íìiîl0-9]\wres', r'exteriores',texto, flags=re.I)
        texto = re.sub('vota[cç][áàaâã]o', r'votação',texto, flags=re.I)
        texto = re.sub('alem[áàaâã]es', r'alemães',texto, flags=re.I)
        texto = re.sub('subs[íìiîl0-9]dio(s?)', r'subsídio\1',texto, flags=re.I)
        texto = re.sub('([^a-z])n[ou]c\wea\w([^a-z])', r'\1nuclear\2',texto, flags=re.I)
        
        texto = re.sub('b[íìiîl0-9]ll[íìiîl0-9]\wns', r'billions',texto, flags=re.I)
        texto = re.sub('manifesta[cç][óòoôõ]es', r'manifestações',texto, flags=re.I)
        texto = re.sub('at[óòoôõ]m[íìiîl0-9]c([ao]s?[^a-z])', r'atômic\1',texto, flags=re.I)
        texto = re.sub('tecnol[óòoôõ]g[íìiîl0-9]c([ao]s?[^a-z])', r'tecnológic\1',texto, flags=re.I)
        texto = re.sub('a[cç][uú]car([^a-z])', r'açúcar\1',texto, flags=re.I)
        texto = re.sub('concess[óòoôõ]es', r'concessões',texto, flags=re.I)
        texto = re.sub('sald[íìiî]var', r'Saldívar',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('conversa[cç][óòoôõ]es', r'conversações',texto, flags=re.I)
        texto = re.sub('amaz[óòoôõ]n[íìiîl0-9]a', r'Amazônia',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('secretarygeneral', r'secretary-general',texto, flags=re.I)
        texto = re.sub('l[íìiîl0-9]berta[cç][áàaâã]o', r'libertação',texto, flags=re.I)
        texto = re.sub('suspens[áàaâã]o', r'suspensão',texto, flags=re.I)
        texto = re.sub('h[íìiîl0-9]drel[èéeê]tric([ao]s?[^a-z])', r'hidrelétric\1',texto, flags=re.I)
        texto = re.sub('revolu[cç][áàaâã]o', r'revolução',texto, flags=re.I)
        texto = re.sub('descolon[íìiîl0-9]za[cç][áàaâã]o', r'descolonização',texto, flags=re.I)
        texto = re.sub('sov[íìiîl0-9][èéeê]tic([ao]s?[^a-z])', r'soviétic\1',texto, flags=re.I)
        texto = re.sub('rod[èéeê]s[íìiîl0-9]a([^a-z])', r'Rodésia\1',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('revo[l1]u[cç][áàaâã]o', r'revolução',texto, flags=re.I)
        texto = re.sub('atl[áàaâã]nt[íìiîl0-9]c([ao]s?)', r'atlântic\1',texto, flags=re.I)
        texto = re.sub('plut[óòoôõ]n[íìiîl0-9]o', r'plutônio',texto, flags=re.I)
        texto = re.sub('subst[íìiîl0-9]tu[íìiîl0-9][cç][áàaâã]o', r'substituição',texto, flags=re.I)
        texto = re.sub('[íìiîl0-9]ta[íìiîl0-9]puh', r'itaipu',texto, flags=re.I)
        texto = re.sub('([^a-z])l[íìiîl0-9]bano([^a-z])', r'\1líbano\2',texto, flags=re.I)
        texto = re.sub('prog\wess?o', r'progresso',texto, flags=re.I)
        texto = re.sub('hem[íìiîl0-9]sf[èéeê]rio', r'hemisfério',texto, flags=re.I)
        texto = re.sub('ant[áàaâã]rt[íìiîl0-9]c([ao]s?[^a-z])', r'antártic\1',texto, flags=re.I)
        texto = re.sub('([^a-z])ameaca([^a-z])', r'\1ameaça\2',texto, flags=re.I)
        texto = re.sub('pol[óòoôõ]n[íìiîl0-9]a', r'Polônia',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('promo[cç][áàaâã]o', r'promoção',texto, flags=re.I)
        texto = re.sub('([^a-z])rom[èéeê]nia([^a-z])', r'\1Romênia\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('([^a-z])n[ou]cl[èéeê]b\w[áàaâã]s([^a-z])', r'\1Nuclebrás\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('d[íìiîl0-9]p[l1]om[áàaâã]t[íìiîl0-9]c([ao]s?[^a-z])', r'diplomátic\1',texto, flags=re.I)
        texto = re.sub('bras[íìiîl0-9]l[èéeê][nñ]([ao]s?[^a-z])', r'brasileñ\1',texto, flags=re.I)
        texto = re.sub('([^\w])asunc[íìiîl0-9][óòoôõ]n([^\w])', r'\1asunción\2',texto, flags=re.I) 
        texto = re.sub('([^\w])[íìiîl0-9]nclus[áàaâã]o([^\w])', r'\1inclusão\2',texto, flags=re.I)
        texto = re.sub('te[l1]egr[áàaâã]fic([ao]s?[^a-z])', r'telegráfic\1',texto, flags=re.I)
        texto = re.sub('organ[íìiîl0-9]zac[íìiîl0-9][óòoôõ]n(e?s?[^a-z])', r'organización',texto, flags=re.I)
        texto = re.sub('plen[íìiîl0-9]potenc[íìiîl0-9][áàaâã]ri([ao]s?[^a-z])', r'plenipotenciári\1',texto, flags=re.I)
        texto = re.sub('([^\w])nogu[èéê]s([^\w])', r'\1Nogués\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('([^\w])voss[èéeê]ncia([^\w])', r'\1vossência\2',texto, flags=re.I)
        texto = re.sub('ass[íìiîl0-9]st[èéeê]nc[íìiîl0-9]a([^li])', r'assistência\1',texto, flags=re.I) #
        texto = re.sub('depe\wde[hnm]', r'dependen',texto, flags=re.I)
        texto = re.sub('l9([0-9][0-9])', r'19\1',texto, flags=re.I)        
        texto = re.sub('nortesu[l1]',r'norte-sul',texto, flags=re.I)
        texto = re.sub('sulafr[íìiîl0-9]c',r'sul-afric',texto, flags=re.I)
        texto = re.sub('agr[íìiîl0-9]co[l1]', r'agrícol',texto, flags=re.I)
        texto = re.sub('navega[cç][áàaâã]o', r'navegação',texto, flags=re.I)
        texto = re.sub('de[l1]egac[íìiî]on', r'delegación',texto, flags=re.I)
        texto = re.sub('[íìiî]ntegrac[íìiî][óòoôõ]n', r'integración',texto, flags=re.I)
        texto = re.sub('organ[íìiî]zac[íìiî][óòoôõ]on', r'organización',texto, flags=re.I)
        texto = re.sub('assoc[íìiî]a[cç][áàaâã]o', r'associação',texto, flags=re.I)
        texto = re.sub('[íìiî]ntegra[cç][áàaâã]o', r'integração',texto, flags=re.I)
        texto = re.sub('conven[cç][áàaâã]o', r'convenção',texto, flags=re.I)
        texto = re.sub('pro[l1][íìiî]fera[cç][áàaâã]o', r'proliferação',texto, flags=re.I)
        
        ######TERMOS DE ALTA RELEVANCIA NO CORPUS TRABALHADO######
        texto = re.sub('([^a-z])n[ou]\wlea\w([^a-z])', r'\1nuclear\2',texto, flags=re.I)
        texto = re.sub('p\w\wít[íìiîl0-9]c([ao]s?[^a-z])', r'polític\1',texto, flags=re.I)
        texto = re.sub('pol\wt\wc([ao]s?[^a-z])', r'polític\1',texto, flags=re.I)
        texto = re.sub('([^a-z][Ss])\w\wve\wra([^a-z])', r'\1ilveira\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('([^a-z][Ss])[íìiîl0-9]\wve[íìiîl0-9]\w\w([^a-z])', r'\1ilveira\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('(silveira)(\w)', r'\1 \2',texto, flags=re.I)
        texto = re.sub('([^\s-])(silveira)', r'\1 \2',texto, flags=re.I)
        texto = re.sub('([^a-z])azere\wo([^a-z])', r'\1Azeredo\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('([^a-z])a\weredo([^a-z])', r'\1Azeredo\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('(azeredo)(\w)', r'\1 \2',texto, flags=re.I)
        texto = re.sub('([^\s-])(azeredo)', r'\1 \2',texto, flags=re.I)
        texto = re.sub('([^a-z])ge[íìiîl0-9]se\w([^a-z])', r'\1Geisel\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('([^\s-])(geisel)', r'\1 \2',texto, flags=re.I)
        texto = re.sub('([^a-z])b\wa[^sz]il([^a-z])', r'\1Brasil\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('([^a-z])b\was\wl([^a-z])', r'\1Brasil\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('([^a-z][Aa])[^\Wr]gola(n?[ao]?s?[^a-z])', r'\1ngola\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('([^a-z][Aa])ngo\wa(n?[ao]?s?[^a-z])', r'\1ngola\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('([^a-z])k\wss\w\wger([^a-z])', r'\1Kissinger\2',texto, flags=re.I)
        texto = re.sub('([^a-z])ki\w\winger([^a-z])', r'\1Kissinger\2',texto, flags=re.I)
        texto = re.sub('([^\s-])k\wss\wnger', r'\1 Kissinger',texto, flags=re.I)
        texto = re.sub('([^a-z])a\wer\wcan([ao]s?[^a-z])', r'\1american\2',texto, flags=re.I)
        texto = re.sub('(\s)(merican)([ao]s?[^a-z])', r'\1a\2\3',texto, flags=re.I)
        texto = re.sub('([^a-z])d\wre[íìiîl0-9]to([^a-z])', r'\1direito\2',texto, flags=re.I)
        texto = re.sub('([^a-z])d[íìiîl0-9]\w?e[íìiîl0-9]\wo([^a-z])', r'\1direito\2',texto, flags=re.I)
        texto = re.sub('([^\s-])(direito)', r'\1 \2',texto, flags=re.I)
        texto = re.sub('d[íìiîl0-9]re[íìiîl0-9]tos[^\s][hmn]umanos', r'direitos humanos',texto, flags=re.I)
        texto = re.sub('[íìiîl0-9]\wdependen', r'independen',texto, flags=re.I)
        texto = re.sub('depe\wdê\wc[íìiîl0-9]a', r'dependência',texto, flags=re.I)
        texto = re.sub('au\wono\w[íìiîl0-9]a', r'autonomia',texto, flags=re.I)
        texto = re.sub('au\wo\wom\wa', r'autonomia',texto, flags=re.I)
        texto = re.sub('não-?[íìiîl0-9]\wte\w\wenç\w\w', r'não-intervenção',texto, flags=re.I)
        texto = re.sub('não-?[íìiîl0-9]nter\wen\wão', r'não-intervenção',texto, flags=re.I)
        texto = re.sub('nac\wonal', r'nacional',texto, flags=re.I)
        texto = re.sub('nac[íìiîl0-9]o\wal', r'nacional',texto, flags=re.I)
        texto = re.sub('dete\wm\wna\wão([^a-z])', r'determinação\1',texto, flags=re.I)
        texto = re.sub('determ[íìiîl0-9]naç\wo([^a-z])', r'determinação \1',texto, flags=re.I)
        texto = re.sub('au\w\w-?determ\w\wa\w\wo', r'auto-determinação',texto, flags=re.I)
        texto = re.sub('su\wve\wsão', r'subversão',texto, flags=re.I)
        texto = re.sub('comu\w[íìiîl0-9]s', r'comunis',texto, flags=re.I)
        texto = re.sub('ant[íìiîl0-9]comu', r'anti-comu',texto, flags=re.I)
        texto = re.sub('g\w[íìiîl0-9]n\w-?b\ws\wau', r'Guiné-Bissau',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('gu\w\wé-?\wissau', r'Guiné-Bissau',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('mo\wamb\w\wue', r'Moçambique',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('go\wbery', r'Golbery',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('\w\wama\waty', r'Itamaraty',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('([^a-z])cr\wmm\wns([^a-z])', r'\1Crimmins\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('([^a-z])d\wt\wnte([^a-z])', r'\1détente\2',texto, flags=re.I)
        texto = re.sub('([^a-z])dé\we\wte([^a-z])', r'\1détente\2',texto, flags=re.I)
        texto = re.sub('([^a-z])Cal\wag\wa\w([^a-z])', r'\1Callaghan\2',texto, flags=re.I) # modificado na versão ignore_case
        texto = re.sub('([^a-z])Ca\wlag\wa\w([^a-z])', r'\1Callaghan\2',texto, flags=re.I) # modificado na versão ignore_case

        ######TERMOS CURTOS, STOPWORDS, PALAVRAS SECUNDÁRIAS######
        texto = re.sub('(\w[\s])aa([\s]\w)',r'\1à\2',texto, flags=re.I)
        texto = re.sub('(\w[\s])aas([\s]\w)',r'\1às\2',texto, flags=re.I)
        texto = re.sub('(\w[\W])jah([\W]\w)',r'\1já\2',texto, flags=re.I)
        texto = re.sub('(\w[\W])oue([\W]\w)',r'\1que\2',texto, flags=re.I)
        texto = re.sub('(\w[\W])t0([\W]\w)',r'\1to\2',texto, flags=re.I)
        texto = re.sub('(\w[\W])estah([\W]\w)',r'\1está\2',texto, flags=re.I)
        texto = re.sub('(\w[\W])ateh([\W]\w)',r'\1até\2',texto, flags=re.I)
        texto = re.sub('([^a-z])tamb[èéeê]m([^a-z])', r'\1também\2',texto, flags=re.I)
        texto = re.sub('a0', r'ao',texto, flags=re.I)
        texto = re.sub('([^\w])al[èéeê]m([^\w])', r'\1além\2',texto, flags=re.I)
        texto = re.sub('([^a-z])por[èéê]m([^a-z])', r'\1porém\2',texto, flags=re.I)
        texto = re.sub('([^a-z])[úùuû]nic([ao]s?[^a-z])', r'\1únic\2',texto, flags=re.I)
        texto = re.sub('([^a-z])tres([^a-z])', r'\1três \2',texto, flags=re.I)
        texto = re.sub('pos[íiìî0-9][cç][áàaâã]o([^a-z])', r'posição\1',texto, flags=re.I)
        texto = re.sub('n[íiìîl0-9]ve[l1]([^\w])', r'nível\1',texto, flags=re.I)
        texto = re.sub('car[áàaâã]ter', r'caráter',texto, flags=re.I)
        texto = re.sub('([^a-z])poss[\w]ve[l0-9]([^a-z])', r'\1possível\2',texto, flags=re.I)
        texto = re.sub('s[íìiîl0-9]tua[çc][áàaâã]o', r'situação',texto, flags=re.I)
        texto = re.sub('dec[íìiîl0-9]s[áàaâã]o', r'decisão',texto, flags=re.I)
        texto = re.sub('atrav[èéeê]s([^a-z])', r'através\1',texto, flags=re.I)
        texto = re.sub('([^a-z])id[èéeê][íìiî]a(s?[^a-z])', r'\1ideia\2',texto, flags=re.I)
        texto = re.sub('([^a-z])pr[óòoôõo]xim([ao]s?[^a-z])', r'\1próxim\2',texto, flags=re.I)
        texto = re.sub('solu[cç][áàaâã]o', r'solução',texto, flags=re.I)
        texto = re.sub('mi[l1]h[óòôoõ]es', r'milhões',texto, flags=re.I)
        texto = re.sub('d[òóoôõ][l1]ares', r'dólares',texto, flags=re.I)
        texto = re.sub('ocas[íìiîl0-9][áàaâã]o', r'ocasião',texto, flags=re.I)
        texto = re.sub('([^a-z])serao([^a-z])', r'\1serão\2',texto, flags=re.I)
        with open(os.path.join(path2,txt), 'w', encoding='utf-8') as f:
            txt = f.write(texto)
        #texto = texto.split()
        #texto = [t.strip(string.punctuation) for t in texto]
        #freqdist = Counter(texto)
        #freqtotal.update(freqdist)
#print(freqtotal.most_common())

sorted_freqtotal = sorted(freqtotal.items(), key=operator.itemgetter(1), reverse=True)

len(sorted_freqtotal)

texto



