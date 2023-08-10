import tensorflow as tf

with tf.Session() as sess:
    add = sess.run( tf.add(4, 40) )
    print( f'Add : {add}' )
    
    subtract = sess.run( tf.subtract(add, 2 ) )
    print( f'Subtract: {subtract}' )
    
    mult = sess.run( tf.multiply(subtract, 3) )
    print( f'Multiply: {mult}' )

x = tf.constant(4.0)
y = tf.constant(2)

with tf.Session() as sess:
    badOperation = sess.run( tf.multiply(x, y))
    print( 'Never here' )

with tf.Session() as sess:
    notSobadOperation = sess.run( tf.multiply( tf.cast(x, tf.int32), y))
    print( f'Everything ok : {notSobadOperation} ' )



