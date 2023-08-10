import theano
from theano import tensor

# Declaração de dois pontos escalares
a = tensor.dscalar()
b = tensor.dscalar()

# Expressão com soma
c = a + b

# Conversão da expressão em objetos callable que recebe os valores (a,b) e calcula o valor de c
f = theano.function([a,b], c)

# Atesta o valor de 1.5 para 'a', e 2.5 para 'b', e avalia o c através da função assert 'c'
assert 4.0 == f(1.5, 2.5)

