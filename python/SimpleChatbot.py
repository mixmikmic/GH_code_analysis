# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Vector을 위한 선언
char_array = [c for c in '[]|abcdef안녕머해누가예뻐어때그래바보']
#금일 수업시간에 나온 enumerate 방법 사용
num_dic = {n: i for i, n in enumerate(char_array)}
[i for i in num_dic]
dic_len = len(num_dic)


#질문에 따른 답변 정의
train_data = [['안녕', 'a'], ['머해', 'b'],
            ['누가', 'c'], ['예뻐', 'd'],
            ['어때', 'e'], ['그래', 'f']]

def make_train_data(train_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in train_data:
        # 인코더 셀의 입력값. 입력단어의 글자들을 한글자씩 떼어 배열로 만든다.
        input = [num_dic[n] for n in seq[0]]
        # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.
        output = [num_dic[n] for n in ('[' + seq[1])]
        # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
        target = [num_dic[n] for n in (seq[1] + ']')]
        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        # 출력값만 one-hot 인코딩이 아님 (sparse_softmax_cross_entropy_with_logits 사용)
        target_batch.append(target)
    return input_batch, output_batch, target_batch

# 옵션 설정
learning_rate = 0.01
n_hidden = 128
total_epoch = 100
# one hot 위한 사이즈 
n_class = n_input = dic_len

#그래프 초기화 (Cell Reuse문제)
tf.reset_default_graph()
# Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같다.
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])

# 인코더 
with tf.variable_scope("encoder"):
    enc_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    #scope.reuse_variables()
    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)


# 디코더 
with tf.variable_scope("decoder"):
    dec_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=0.5)
    # Seq2Seq 모델 구현
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states,
                                            dtype=tf.float32)

model = tf.layers.dense(outputs, n_class, activation=None)

cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
input_batch, output_batch, target_batch = make_train_data(train_data)
print('<학습결과 Plot 출력>')
plot_X = []
plot_Y = []
for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={enc_input: input_batch,
                                  dec_input: output_batch,
                                  targets: target_batch})
    plot_X.append(epoch + 1)
    plot_Y.append (loss)                       
# Graphic display
plt.plot(plot_X, plot_Y, label='cost')
plt.show()
 
# 최적화가 끝난 뒤, 변수를 저장합니다.
saver = tf.train.Saver(tf.global_variables())
saver.save(sess, ''.join("./model"))

# 단어를 입력받아 번역 단어를 예측하고 디코딩하는 함수
def predict(word):
    input_batch, output_batch, target_batch = make_train_data([word])
    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
    prediction = tf.argmax(model, 2)
    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})
    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = char_array[result[0][0]]
    return decoded

def response(decode):
    decode = predict(decode)
    if(decode == 'a') :
        return "반가워 봇"
    elif(decode == 'b') :
        return "Tensorflow 수업듣고 있어"
    elif(decode == 'c') :
        return "남중구 선생님한테 배우고 있어"
    elif(decode == 'd') :
        return "남자 선생님이야;;"
    elif(decode == 'e') :
        return "잘가르쳐 주시고 많이 배우고 있어"
    elif(decode == 'f') :
        return "그래 안녕~"
    else :
        return "무슨말인지 모르겠다"

print('나 : 안녕')
print('봇 :', response('안녕'))

print('\n나 : 머해')
print('봇 :', response('머해'))

print('\n나 : 누가')
print('봇 :', response('누가'))

print('\n나 : 예뻐')
print('봇 :', response('예뻐'))

print('\n나 : 어때')
print('봇 :', response('어때'))

print('\n나 : 바보')
print('봇 :', response('바보'))

print('\n나 : 그래')
print('봇 :', response('그래'))




