from gensim.models import KeyedVectors
from debiaswe.we import WordEmbedding

E = WordEmbedding('./embeddings/cbow_s300.txt')

#model = KeyedVectors.load_word2vec_format('./embeddings/cbow_s300.txt', unicode_errors="ignore")
tmp_file='./data/temp.bin'
E.save_w2v(tmp_file)
print ("saved")
from gensim.models import KeyedVectors
tmp_file='./data/temp.bin'
word_vectors = KeyedVectors.load_word2vec_format(tmp_file, binary=True)
#cbow_s300.txt was obtained from http://www.nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc

#analogies_list = ['programador', 'maestro', 'capitão', 'favorecido', 'filósofo', 'capitão', 'arquiteto',
#                  'financista', 'batalhador', 'radiodifusor', 'mágico']
analogies_list = ['advogado', 'arquiteto', 'ator', 'bibliotecario', 'biologo', 'blogueiro', 'bombeiro', 'cabeleireiro', 'camareiro', 'cantor', 'consultor', 'contador', 'coordenador', 'coreografo', 'costureiro', 'cozinheiro', 'dançarino', 'decorador', 'diagramador', 'diretor', 'empresario', 'enfermeiro', 'engenheiro', 'escritor', 'estagiario', 'fotógrafo', 'garçom', 'historiador', 'instrutor', 'juiz', 'matemático', 'médico', 'pedagogo', 'pesquisador', 'pintor', 'professor', 'programador', 'promotor', 'psicopedagogo', 'psicologo', 'publicitario', 'secretario', 'senador', 'sociologo', 'sindico', 'tutor', 'vendedor', 'vereador', 'veterinario']

"""for i in analogies_list:
    x = model.wv.most_similar(positive=['mulher', i], negative=['homem'])
    print(x[0])
""" 
extreme_she = []
for x in analogies_list:
    if (x in word_vectors.vocab):
        k = word_vectors.most_similar(positive=['ela', x], negative=['ele'])[0] 
        extreme_she.append((x,k[1],k[0]))
        
extreme_she.sort(key=lambda tup: tup[1])
for x in reversed(extreme_she[-10:]):
    print(x)

analogies_list2 = ['advogada', 'arquiteta', 'atriz', 'bibliotecaria', 'biologa', 'blogueira', 'bombeira', 'cabeleireira', 'camareira', 'cantora', 'consultora', 'contadora', 'coordenadora', 'coreografa', 'costureira', 'cozinheira', 'dançarina', 'decoradora', 'diagramadora', 'diretora', 'empresaria', 'enfermeira', 'engenheira', 'escritora', 'estagiaria', 'fotógrafa', 'garçonete', 'historiadora', 'instrutora', 'juiza', 'matemática', 'médica', 'pedagoga', 'pesquisadora', 'pintora', 'professora', 'programadora' 'promotora', 'psicopedagoga', 'psicologa', 'publicitaria', 'secretaria', 'senadora', 'sociologa', 'sindica', 'tutora', 'vendedora', 'vereadora', 'veterinaria']
#analogies_list2 = ['empregada', 'enfermeira', 'recepcionista', 'bibliotecário', 'socialite', 'cabeleireiro',
#                  'babá', 'bibliotecária', 'estilista', 'governanta']

"""
for i in analogies_list2:
    x = model.wv.most_similar(positive=['homem', i], negative=['mulher'])
    print(x[0])
"""
extreme_he = []
for x in analogies_list2:
    if (x in word_vectors.vocab):
        k = word_vectors.most_similar(positive=['ela', x], negative=['ele'])[0]
        extreme_he.append((x,k[1],k[0]))
        
extreme_he.sort(key=lambda tup: tup[1])
for x in reversed(extreme_he[-10:]):
    print(x)

analogies_list3 = ['costureira', 'enfermeira', 'empregada', 'designer', 'softball',
                   'loiro', 'feminismo', 'cosméticos', 'risadinha', 'vocalista', 'pequena', 'atrevido',
                   'divã', 'encantador', 'voleibol', 'cupcake', 'adorável']

for i in analogies_list3:
    x = model.wv.most_similar(positive=['mulher', i], negative=['homem'])
    print(x[0])

analogies_list4 = ['carpintaria', 'médico', 'lojista', 'cirurgião', 'arquiteto', 'beisebol', 'corpulento',
                   'conservadorismo', 'fármacos', 'risada', 'guitarrista', 'esguio', 'mal-humorado', 'famoso',
                   'afável', 'futebol', 'pizzas', 'brilhante']

for i in analogies_list4:
    x = model.wv.most_similar(positive=['homem', i], negative=['mulher'])
    print(x[0])

