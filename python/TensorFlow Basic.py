import tensorflow as tf

# 1x2 행렬을 만드는 constant op을 생성
# 이 op는 default graph에 노드로 들어갈 것
# 생성함수에서 나온 값은 constant op의 결과 값
matrix1 = tf.constant([[3., 3.]])

# 1x2 행렬을 만드는 constant op를 생성
matrix2 = tf.constant([[2.],[2.]])

# 'matrix1'과 'matrix2를 입력값으로 하는 Matmul op(행렬 곱)
# 이 op의 결과값인 'product'는 행렬곱의 결과를 의미
product = tf.matmul(matrix1, matrix2)

print(product)

# default graph를 실행
sess = tf.Session()

# 행렬곱 작업(op)을 실행하기 위해 session의 'run()' 메서드를 호출해서 행렬곱 
# 작업의 결과값인 'product' 값을 넘겨줍시다. 그 결과값을 원한다는 뜻
#
# 작업에 필요한 모든 입력값들은 자동적으로 session에서 실행되며 보통은 병렬로 처리
#
# 'run(product)'가 호출되면 op 3개가 실행, 2개는 상수고 1개는 행렬 곱
#
# 작업의 결과물은 numpy `ndarray` 오브젝트인 result' 값
result = sess.run(product)
print(result)

# 실행을 마치면 Session을 닫기
sess.close()

