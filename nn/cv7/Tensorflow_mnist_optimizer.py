import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# train params
training_epochs = 15
batch_size = 100
display_step = 50
learning_rate = 0.001

# network params
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_classes = 10

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

def Model_NN_with_batch(x, weights, biases):
    with tf.name_scope('Model_NN'):
        #Hidden Layer 1
        layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['h1'])
        layer_1 = tf.contrib.layers.batch_norm(layer_1, decay =0.9, scope = 'Layer_1_bn', activation_fn = tf.nn.relu)
        layer_1 = tf.nn.relu(layer_1)

        # Hidden Layer 2
        layer_2= tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
        layer_2 = tf.contrib.layers.batch_norm(layer_2, decay=0.9, scope='Layer_2_bn', activation_fn=tf.nn.relu)
        layer_2 = tf.nn.relu(layer_2)

        # Output Layer
        out_layer = tf.add(tf.matmul(layer_2,weights['out']),biases['out'])

        return out_layer


def Model_NN_without_batch(x, weights, biases):
    with tf.name_scope('Model_NN'):
        #Hidden Layer 1
        layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['h1'])
      # layer_1 = tf.contrib.layers.batch_norm(layer_1, decay =0.9, scope = 'Layer_1_bn', activation_fn = tf.nn.relu)
        layer_1 = tf.nn.relu(layer_1)

        # Hidden Layer 2
        layer_2= tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
     #  layer_2 = tf.contrib.layers.batch_norm(layer_2, decay=0.9, scope='Layer_2_bn', activation_fn=tf.nn.relu)
        layer_2 = tf.nn.relu(layer_2)

        # Output Layer
        out_layer = tf.add(tf.matmul(layer_2,weights['out']),biases['out'])

        return out_layer

class Net:
    def __init__(self, optimizer, opt,**kwargs):
        tf.reset_default_graph()

        self.weights = {'h1': tf.get_variable(name='w1_xavier', shape=[n_input, n_hidden_1],
                                         initializer=tf.contrib.layers.xavier_initializer()),
                   'h2': tf.get_variable(name='w2_xavier', shape=[n_hidden_1, n_hidden_2],
                                         initializer=tf.contrib.layers.xavier_initializer()),
                   'out': tf.get_variable(name='wout_xavier', shape=[n_hidden_2, n_classes],
                                          initializer=tf.contrib.layers.xavier_initializer())}

        self.biases = {'h1': tf.Variable(tf.random_normal(shape=[n_hidden_1], name='b1_normal')),
                  'h2': tf.Variable(tf.random_normal(shape=[n_hidden_2], name='b2_normal')),
                  'out': tf.Variable(tf.random_normal(shape=[n_classes], name='bout_normal'))}
    #placeholder
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])

    #model
        self.y_prediction = Model_NN_with_batch(self.x, self.weights, self.biases)

    #loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_prediction,labels=self.y))

    #accuracy
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_prediction, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.train_step = opt(learning_rate=learning_rate, **kwargs).minimize(self.loss)

    #writer
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('logs/'+optimizer)

def main(batch = True,optimizer = 'SGD'):
    tf.reset_default_graph()

    if optimizer == 'SGD':
        net = Net(optimizer = optimizer, opt = tf.train.GradientDescentOptimizer)
    elif optimizer == 'momentum':
        net = Net(optimizer = optimizer, opt = tf.train.MomentumOptimizer, momentum=0.9)
    elif optimizer == 'RMSPropOptimizer':
        net = Net(optimizer = optimizer, opt = tf.train.RMSPropOptimizer)
    elif optimizer == 'Adam':
        net = Net(optimizer = optimizer, opt = tf.train.AdamOptimizer)

    #tensorflow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


    #Tensorborad writer
        # if batch:
        #     writer = tf.summary.FileWriter('logs/withbatch')
        # else:
        #     writer = tf.summary.FileWriter('logs/withoutbatch',sess.graph)

    #loop for epoches:

        for epoch in range(training_epochs):

            for i in range(int(mnist.train.num_examples/batch_size)):
                batch = mnist.train.next_batch(batch_size)

                _, train_accuracy, train_summary = sess.run([net.train_step, net.accuracy, net.summary], feed_dict={net.x: batch[0], net.y: batch[1]})

                net.writer.add_summary(train_summary,epoch*batch_size+i)

                if i % display_step ==0:
                    print("Epoch: %04d, Optimizer:%s, Batch_Index: %4d/%4d, training_accuracy: %.5f" %(epoch+1,optimizer,i,int(mnist.train.num_examples/batch_size),train_accuracy) )
            test_accuracy, = sess.run(fetches = [net.accuracy,],feed_dict = {net.x:mnist.test.images, net.y:mnist.test.labels})
            print("Epoch: %04d, Test_Accuracy is %.5f"%(epoch +1,test_accuracy))
    # if batch:
    #     tf.gfile.DeleteRecursively('logs/withbatch')
    # else:
    #     tf.gfile.DeleteRecursively('logs/withoutbatch')

if __name__ == '__main__':
    #main(batch=False)
    main(batch=True,optimizer='SGD')
    main(batch=True, optimizer='momentum')
    main(batch=True, optimizer = 'RMSPropOptimizer')
    main(batch=True, optimizer = 'Adam')