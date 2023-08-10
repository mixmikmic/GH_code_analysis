import pandas as pd
import glob
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
matplotlib.rcParams['svg.fonttype'] = 'none'

files = glob.glob(r'..\data\consulta_cand_2014\consulta_cand_2014_*.txt')
candidates = pd.concat([pd.read_csv(file, encoding='Latin1', sep=';', header=None,error_bad_lines=False,warn_bad_lines=True) for file in files])
candidates.columns =["DATA_GERACAO","HORA_GERACAO","ANO_ELEICAO","NUM_TURNO","DESCRICAO_ELEICAO","SIGLA_UF","SIGLA_UE","DESCRICAO_UE","CODIGO_CARGO","DESCRICAO_CARGO","NOME_CANDIDATO","SEQUENCIAL_CANDIDATO","NUMERO_CANDIDATO","CPF_CANDIDATO","NOME_URNA_CANDIDATO","COD_SITUACAO_CANDIDATURA","DES_SITUACAO_CANDIDATURA","NUMERO_PARTIDO","SIGLA_PARTIDO","NOME_PARTIDO","CODIGO_LEGENDA","SIGLA_LEGENDA","COMPOSICAO_LEGENDA","NOME_LEGENDA","CODIGO_OCUPACAO","DESCRICAO_OCUPACAO","DATA_NASCIMENTO","NUM_TITULO_ELEITORAL_CANDIDATO","IDADE_DATA_ELEICAO","CODIGO_SEXO","DESCRICAO_SEXO","COD_GRAU_INSTRUCAO","DESCRICAO_GRAU_INSTRUCAO","CODIGO_ESTADO_CIVIL","DESCRICAO_ESTADO_CIVIL","CODIGO_COR_RACA","DESCRICAO_COR_RACA","CODIGO_NACIONALIDADE","DESCRICAO_NACIONALIDADE","SIGLA_UF_NASCIMENTO","CODIGO_MUNICIPIO_NASCIMENTO","NOME_MUNICIPIO_NASCIMENTO","DESPESA_MAX_CAMPANHA","COD_SIT_TOT_TURNO","DESC_SIT_TOT_TURNO","NM_EMAIL",]
candidates = candidates[["SIGLA_UF","SIGLA_UE","CODIGO_CARGO","DESCRICAO_CARGO","SEQUENCIAL_CANDIDATO","NOME_URNA_CANDIDATO","DES_SITUACAO_CANDIDATURA","NUMERO_PARTIDO","SIGLA_PARTIDO","DESCRICAO_OCUPACAO","DATA_NASCIMENTO","CODIGO_SEXO","DESCRICAO_SEXO","DESCRICAO_GRAU_INSTRUCAO","DESCRICAO_COR_RACA","SIGLA_UF_NASCIMENTO","CODIGO_MUNICIPIO_NASCIMENTO","COD_SIT_TOT_TURNO","DESC_SIT_TOT_TURNO","DESPESA_MAX_CAMPANHA","DATA_NASCIMENTO"]]
candidates.head(1)

candidates.shape

nominal_votes.shape

files = glob.glob(r'..\\data\\votacao_candidato_munzona_2014\\votacao_candidato_munzona_2014_*.txt')
nominal_votes = pd.concat([pd.read_csv(file, encoding='Latin1', sep=';', header=None,error_bad_lines=False,warn_bad_lines=True) for file in files])
nominal_votes.columns=["DATA_GERACAO","HORA_GERACAO","ANO_ELEICAO","NUM_TURNO","DESCRICAO_ELEICAO","SIGLA_UF","SIGLA_UE","CODIGO_MUNICIPIO","NOME_MUNICIPIO","NUMERO_ZONA","CODIGO_CARGO","NUMERO_CAND","SQ_CANDIDATO","NOME_CANDIDATO","NOME_URNA_CANDIDATO","DESCRICAO_CARGO","COD_SIT_CAND_SUPERIOR","DESC_SIT_CAND_SUPERIOR","CODIGO_SIT_CANDIDATO","DESC_SIT_CANDIDATO","CODIGO_SIT_CAND_TOT","DESC_SIT_CAND_TOT","NUMERO_PARTIDO","SIGLA_PARTIDO","NOME_PARTIDO","SEQUENCIAL_LEGENDA","NOME_COLIGACAO","COMPOSICAO_LEGENDA","TOTAL_VOTOS","TRANSITO",]
nominal_votes = nominal_votes[["SIGLA_UF","SIGLA_UE","CODIGO_MUNICIPIO","NOME_MUNICIPIO","SQ_CANDIDATO","NOME_URNA_CANDIDATO","DESCRICAO_CARGO","SIGLA_PARTIDO","TOTAL_VOTOS"]]
nominal_votes.head(1)

files = glob.glob(r'..\\data\\votacao_partido_munzona_2014\\votacao_partido_munzona_2014_*.txt')
party_votes = pd.concat([pd.read_csv(file, encoding='Latin1', sep=';', header=None,error_bad_lines=False,warn_bad_lines=True) for file in files])
party_votes.columns=["DATA_GERACAO","HORA_GERACAO","ANO_ELEICAO","NUM_TURNO","DESCRICAO_ELEICAO","SIGLA_UF","SIGLA_UE","CODIGO_MUNICIPIO","NOME_MUNICIPIO","NUMERO_ZONA","CODIGO_CARGO","DESCRICAO_CARGO","TIPO_LEGENDA","NOME_COLIGACAO","COMPOSICAO_LEGENDA","SIGLA_PARTIDO","NUMERO_PARTIDO","NOME_PARTIDO","QTDE_VOTOS_NOMINAIS","QTDE_VOTOS_LEGENDA","TRANSITO","SEQUENCIAL_COLIGACAO",]
party_votes = party_votes[["SIGLA_UF","SIGLA_UE","CODIGO_MUNICIPIO","DESCRICAO_CARGO","TIPO_LEGENDA","NOME_COLIGACAO","COMPOSICAO_LEGENDA","SIGLA_PARTIDO","QTDE_VOTOS_NOMINAIS","QTDE_VOTOS_LEGENDA","SEQUENCIAL_COLIGACAO",]]
party_votes.head(1)

files = glob.glob(r'..\\data\bem_candidato_2014\\bem_candidato_2014_*.txt')
possessions = pd.concat([pd.read_csv(file, encoding='Latin1', sep=';', header=None,error_bad_lines=False,warn_bad_lines=True) for file in files])
possessions.columns=["DATA_GERACAO","HORA_GERACAO","ANO_ELEICAO","DESCRICAO_ELEICAO","SIGLA_UF","SQ_CANDIDATO","CD_TIPO_BEM_CANDIDATO","DS_TIPO_BEM_CANDIDATO","DETALHE_BEM","VALOR_BEM","DATA_ULTIMA_ATUALIZACAO","HORA_ULTIMA_ATUALIZACAO",]
possessions = possessions[["SQ_CANDIDATO","CD_TIPO_BEM_CANDIDATO","DS_TIPO_BEM_CANDIDATO","DETALHE_BEM","VALOR_BEM",]]
possessions.head(1)

camara_hj = candidates[(candidates.DESCRICAO_CARGO=='DEPUTADO FEDERAL') & ((candidates.DESC_SIT_TOT_TURNO=='ELEITO POR QP')|(candidates.DESC_SIT_TOT_TURNO=='ELEITO POR MÉDIA'))]

# Merging the candidates datagrame with the sum of all votes by each candidate
candidates = candidates.merge((nominal_votes.groupby(['SQ_CANDIDATO'])['TOTAL_VOTOS'].sum().to_frame().reset_index()),how='left',left_on='SEQUENCIAL_CANDIDATO',right_on='SQ_CANDIDATO')

# Keeping only candidates to the Câmara dos Deputados
candidates_federal = candidates[candidates.DESCRICAO_CARGO=='DEPUTADO FEDERAL']

states = ['AC','AL','AP','AM','BA','CE','DF','ES','GO','MA','MT','MS','MG','PR','PB','PA','PE','PI','RJ','RN','RS','RO','RR','SC','SE','SP','TO',]

bancadas = {} # Dictionary where I will save each states representatives
for state in states:
    count = camara_hj[camara_hj.SIGLA_UE==state]['SEQUENCIAL_CANDIDATO'].count() # Number of chairs each state had in 2014
    df = candidates_federal[candidates_federal.SIGLA_UE==state].sort_values(by='TOTAL_VOTOS',ascending=False).head(count) # Get the most voted candidates for each state
    bancadas[state] = df

# Concatenating all the dataframes
camara_distritao = pd.concat([bancadas[state] for state in states]) 

# How many of the candidates in the distritão datafram
camara_distritao.SEQUENCIAL_CANDIDATO.isin(camara_hj.SEQUENCIAL_CANDIDATO).value_counts()

# Keeping only candidates to the Assembléia Legislativa - state level Congress
candidates_assembleia = candidates[(candidates.DESCRICAO_CARGO=='DEPUTADO ESTADUAL')|(candidates.DESCRICAO_CARGO=='DEPUTADO DISTRITAL')]

# Getting the composition of each state and storing the dataframe in a dictionary
assembleias_hj = {}
for state in states:
    df = candidates_assembleia[(candidates_assembleia.SIGLA_UE==state)&((candidates.DESC_SIT_TOT_TURNO=='ELEITO POR QP')|(candidates.DESC_SIT_TOT_TURNO=='ELEITO POR MÉDIA'))]
    assembleias_hj[state] = df

# Getting the number of representatives for each state
assembleias_distritao = {}
for state in states:
    count = candidates_assembleia[(candidates_assembleia.SIGLA_UE==state)&((candidates.DESC_SIT_TOT_TURNO=='ELEITO POR QP')|(candidates.DESC_SIT_TOT_TURNO=='ELEITO POR MÉDIA'))]['SEQUENCIAL_CANDIDATO'].count()
    df = candidates_assembleia[candidates_assembleia.SIGLA_UE==state].sort_values(by='TOTAL_VOTOS',ascending=False).head(count)
    assembleias_distritao[state] = df

all_assembleias = []
for state in states:
    df = assembleias_distritao[state]['SEQUENCIAL_CANDIDATO'].isin(assembleias_hj[state]['SEQUENCIAL_CANDIDATO']).value_counts(normalize=False).to_frame()
    df.columns = [state]
    df = df.transpose()
    df.columns = ['would_stay','would_change']
    df['total_representatives'] = df['would_stay'] + df['would_change']
    df['would_stay_%'] = df['would_stay']/df['total_representatives']
    df['would_change_%'] = (df['would_change']/df['total_representatives'])*100
    df = df.reset_index()
    df = df.rename(columns={'index':'state'})
    all_assembleias.append(df)

change_states = pd.concat(all_assembleias,ignore_index=True)

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(11,16))
sns.stripplot(data=change_states.sort_values(by='would_change_%',ascending=False),
                   y='state',
                   x='would_change_%',
                   color='red',
                   size=14,)

# Plotting a line with the simulated changes in National Congress
x1 = 8.7719 # Percentage of change in National Congress
y1 = 26 # Ending position, in the top of the chart - arbitrary, since Y values are categorical, not numerical
x2 = 8.7719
y2 = 0 # Starting position, in the bottom of the chart - arbitrary, since Y values are categorical, not numerical
plt.plot([x1,x2],[y1,y2],color='black',linestyle=':') # Plotting the line I've just set


# Setting a custom font for the labels
custom_font = {'fontname':'Gill Sans MT','size':'20'}

ax.tick_params(labelsize=15)
ax.set_xlim(0)
ax.set_ylabel('',**custom_font)
ax.set_xlabel('Percentage of representatives that would change if reform was in place on the 2014 Election',**custom_font)
ax.yaxis.grid(True)
ax.xaxis.grid(False)
sns.despine(left=True,bottom=True)

plt.savefig(r'..\visuals\states_change.svg',transparent=True)

for state in states:
    print(state)
    print('Hoje:')
    print(assembleias_hj[state]['SIGLA_PARTIDO'].value_counts().count())
    print('Projeção:')
    print(assembleias_distritao[state]['SIGLA_PARTIDO'].value_counts().count())
    print()

change_states.columns = ['estado','deputados_ficariam','deputados_mudariam','total_deputados','deputados_ficariam_%','deputados_mudariam_%']

change_states = change_states[['estado','deputados_mudariam_%']]

change_states.to_csv('mudancas-assembleias-legislativas.csv',index=False)

assembleias_distritao['PR']['SIGLA_PARTIDO'].value_counts()

assembleias_hj['PR']['SIGLA_PARTIDO'].value_counts()



