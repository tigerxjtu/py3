import tensorflow as tf

a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
c=tf.placeholder(tf.float32)

p1=tf.subtract(0.0,b)

p21=tf.square(b)
p22=tf.multiply(tf.multiply(4.0,a),c)
p2=tf.sqrt(tf.subtract(p21,p22))

p3=tf.multiply(2.0,a)

x1=tf.div(tf.add(p1,p2),p3)
x2=tf.div(tf.subtract(p1,p2),p3)

with tf.Session() as sess:
    result=sess.run([x1,x2],feed_dict={a : 2, b : 7, c : 4})
    print('result:',result)