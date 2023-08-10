import pandas as pd
import glob

# Glob get's a list of all the .csv files in the directory.
files = glob.glob(r'C:\Users\Avell\Desktop\the-lede-program\data-studio\government-credit-cards\files\*.csv')

# Dataframe made out of the first file
df = pd.read_csv(r'.\files\201001_CPGF.csv', encoding='Latin5',sep='\t')

# Concatenating loop
df = pd.concat([pd.read_csv(file, encoding='Latin5', sep='\t', header=None, skiprows=1,error_bad_lines=False,warn_bad_lines=True) for file in files])

#Since I skipped the headers, I'll add them manually:
df.columns = ['Código Órgão Superior','Nome Órgão Superior','Código Órgão Subordinado','Nome Órgão Subordinado','Código Unidade Gestora','Nome Unidade Gestora','Ano Extrato','Mês Extrato','CPF Portador','Nome Portador','Transação','Data Transação','CNPJ ou CPF Favorecido','Nome Favorecido','Valor Transação']

df.to_csv('CPGF_2010_to_2017_july.csv',index=False)

