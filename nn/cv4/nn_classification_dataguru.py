import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy
from sklearn import svm,linear_model

############prepare the data
data_scale = 100
cov_mat1 = np.array([[0,1],[1,0]])

mu_vec1 = np.array([-1,0])
x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, data_scale)
mu_vec1 = mu_vec1.reshape(1,2).T # to 1-col vector

mu_vec2 = np.array([1,4])
cov_mat2 = np.array([[2,-1],[0,1]])
x2_samples = np.random.multivariate_normal(mu_vec2, cov_mat2, data_scale)
mu_vec2 = mu_vec2.reshape(1,2).T




# plt.scatter(x3_samples[:,0],x3_samples[:,1], c= 'red', marker='*')

X = np.concatenate((x1_samples,
                    x2_samples,
                    # x3_samples
                    ), axis = 0)
Y = np.array([0]*data_scale + [1]*data_scale
                    # + [1]*100
                    )





###############prediction

def predict(model,x):
    W1,b1,W2,b2 = model['W1'],model['b1'],model['W2'],model['b2']
    # forwarding
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_score = np.exp(z2)
    probs = exp_score / np.sum(exp_score,axis=1,keepdims=True)
    return np.argmax(probs,axis=1)

###############plot unlinear boundary
def unlinearplotallpoints(model):

    all_points = []
    for i in np.arange(-6, 6, .05):
        for j in np.arange(-3, 8, .05):
            all_points.append([i,j])
    points = np.array(all_points)

    predicts = predict(model,points)

    plt.scatter(points[...,0],points[...,1], c=predicts, marker='1',cmap='Blues',alpha=.1)
    # plt.scatter(nppoints[...,0],nppoints[...,1], c=npprobs[...,1], marker='1',cmap='Blues',alpha=alpha)

###############

training_scale = X.shape[0]
nn_input_dim = 2
nn_output_dim = 2
epsilon = 0.01


#######

def calulate_loss (model):
    W1,b1,W2,b2 = model['W1'],model['b1'],model['W2'],model['b2']
    # forwarding
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_score = np.exp(z2)
    probs = exp_score / np.sum(exp_score,axis=1,keepdims=True)

    # Calculating the loss
    correct_logprobs = -np.log(probs[range(training_scale),Y])
    data_loss = np.sum(correct_logprobs)

    # Add a regulatization term to loss
 #   data_loss += reg_lamda/ 2 * (np.sum(np.square(W1))) * (np.sum(np.square((W2))))

    return data_loss

def build_model (nn_hidden, number_of_passes = 1, print_loss = False):
    # initiate the W and b

    W1 = np.random.randn(nn_input_dim,nn_hidden) / np.square(nn_output_dim)
    b1 = np.zeros((1,nn_hidden))
    W2 = np.random.randn(nn_hidden,nn_output_dim) / np.square(nn_hidden)
    b2 = np.zeros((1, nn_output_dim))

    model = {}

    # Gradient descent
    for i in range(0, number_of_passes):
        # forwarding
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_score = np.exp(z2)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)

        # backwards
        delta3 = probs
        delta3[range(training_scale),Y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3,axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1-np.power(a1,2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2,axis=0)

        # add regularization
        # dW2 += reg_lamda * W2
        # dW1 += reg_lamda * W1

        # refresh W and b
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2


        model = {'W1':W1, 'b1':b1, 'W2':W2, 'b2': b2}

        if print_loss and i%1 == 0:
            plt.scatter(x1_samples[:, 0], x1_samples[:, 1], c='red', marker='*')
            plt.scatter(x2_samples[:, 0], x2_samples[:, 1], c='green', marker='o')

            showstring = 'iteration:'+str(i)+',loss:'+str(calulate_loss(model))
            print(showstring)

    unlinearplotallpoints(model)
    plt.text(-6, -3, showstring)
    plt.show()


    return model



model = build_model(nn_hidden=14,number_of_passes=400,print_loss=True)







#plotboundary(b.coef_[0],b.intercept_[0],'r--')

