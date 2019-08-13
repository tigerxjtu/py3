import numpy as np
import tensorflow as tf


data_scale = 100
cov_mat1 = np.array([[0,1],[1,0]])

mu_vec1 = np.array([-1,0])
x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, data_scale)
mu_vec1 = mu_vec1.reshape(1,2).T # to 1-col vector

mu_vec2 = np.array([1,4])
cov_mat2 = np.array([[2,-1],[0,1]])
x2_samples = np.random.multivariate_normal(mu_vec2, cov_mat2, data_scale)
mu_vec2 = mu_vec2.reshape(1,2).T



X = np.concatenate((x1_samples,
                    x2_samples,
                    ), axis = 0)
Y = np.array([[1,0]]*data_scale + [[0,1]]*data_scale
                    )
num_examples=200
shuffle_indices = np.random.permutation(np.arange(num_examples))
X=X[shuffle_indices]
Y=Y[shuffle_indices]

# train params
training_epochs = 10
batch_size = 20
display_step = 5
learning_rate = 0.01



# network params
n_input = 2
n_hidden_1 = 4
n_hidden_2 = 4
n_classes = 2

def Model_NN(x, weights, biases):
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

def predict(y_prediction):
    x = tf.placeholder(tf.float32,[None, n_input])
    y = tf.placeholder(tf.float32,[None, n_classes])
    with tf.name_scope("prediction_accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y, 1))
        prediction_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return prediction_accuracy


def main():
    #initialization

    weights = { 'h1' : tf.get_variable(name = 'w1_xavier',shape = [n_input,n_hidden_1], initializer= tf.contrib.layers.xavier_initializer()),
                'h2' : tf.get_variable(name = 'w2_xavier',shape = [n_hidden_1,n_hidden_2], initializer= tf.contrib.layers.xavier_initializer()),
                'out': tf.get_variable(name='wout_xavier', shape=[n_hidden_2, n_classes],initializer=tf.contrib.layers.xavier_initializer())}

    biases = {   'h1' : tf.Variable(tf.random_normal(shape=[n_hidden_1],name='b1_normal')),
                'h2' : tf.Variable(tf.random_normal(shape=[n_hidden_2],name='b2_normal')),
                'out' : tf.Variable(tf.random_normal(shape=[n_classes],name='bout_normal'))}



    #input data

    x = tf.placeholder(tf.float32,[None, n_input])
    y = tf.placeholder(tf.float32,[None,n_classes])

    #model

    y_prediction = Model_NN(x, weights, biases)

    #Loss Function
    with tf.name_scope('Loss'):
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_prediction,labels=y))
        loss = tf.losses.mean_squared_error(y_prediction,y)
    #Optimizer & backwards:
    with tf.name_scope('Gradient_Descent'):
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    #Get Accuracy of model:
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_prediction,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #Tensorboard

    tf.summary.scalar('loss',loss)
    tf.summary.scalar('accuracy',accuracy)
    summary = tf.summary.merge_all()

    #tensorflow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Tensorborad writer
        writer = tf.summary.FileWriter('logs', sess.graph)

        # loop for epoches:

        for epoch in range(training_epochs):

            for i in range(int(num_examples / batch_size)):
                # print(batch)
                _, train_accuracy, train_summary = sess.run([train_step, accuracy, summary],
                                                            feed_dict={x: X[i*batch_size:(i+1)*batch_size],
                                                                       y: Y[i*batch_size:(i+1)*batch_size]})
                # _, train_accuracy, train_summary = sess.run([train_step, accuracy, summary],
                #                                             feed_dict={x: X[i:i + batch_size], y: Y[i:i + batch_size]})

                writer.add_summary(train_summary, epoch * batch_size + i)

                if i % display_step == 0:
                    print("Epoch: %04d, Batch_Index: %4d/%4d, training_accuracy: %.5f" % (
                    epoch + 1, i, int(num_examples / batch_size), train_accuracy))


if __name__ == '__main__':
    main()
