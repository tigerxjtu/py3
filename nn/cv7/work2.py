import tensorflow as tf
def model(learning_rate):
    x = tf.Variable(20.0)
    sq = x ** 2
    o = tf.train.GradientDescentOptimizer(learning_rate).minimize(sq)
    return x, o

def main(learning_rate):
    with tf.Session() as sess:
        x, o = model(learning_rate)
        init = tf.global_variables_initializer()
        sess.run([init])

        for i in range(101):
            sess.run([o])
        (x,) = sess.run([x])
        print(x)

rates=[0.0001,0.01,1.0,1.1]
for r in rates:
    tf.reset_default_graph()
    main(r)
