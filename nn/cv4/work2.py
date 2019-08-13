import numpy as np

def softmax(data):
    exp_score = np.exp(data)
    probs = exp_score / np.sum(exp_score, axis=0, keepdims=True)
    return probs


def softmax_crossentropy_loss(data,expects):
    probs=softmax(data)
    correct_logprobs = -np.log(probs[expects==1])
    data_loss = np.sum(correct_logprobs)
    return data_loss

def hinge_loss_record(record,tag,delta):
    correct_value=record[tag==1][0]
    losses=0
    for i,v in enumerate(record):
        if tag[i]==1:
            continue
        loss=max(0,(v-correct_value+delta))
        losses+=loss
    return losses

def hinge_loss(data,expects,delta):
    losses=0
    for i in range(data.shape[1]):
        loss=hinge_loss_record(data[:,i].flatten(),expects[:,i].flatten(),delta)
        losses += loss
    return losses


data=np.array([[7.07,0.62,8.19,1,8.19,3.84],
               [4.54,2.86,3.5,3.61,3.54,9.82],
               [3.21,7.03,9.52,9.52,3.14,7.52],
               [7.81,6.21,9.58,2.57,3.23,9.82]])
corrects=np.zeros(data.shape,dtype=np.int)
labels=[0,3,0,1,0,1]
for i,v in enumerate(labels):
    corrects[v,i]=1

# probs=softmax(data)
# correct_logprobs = -np.log(probs[corrects==1])
#
# print(probs)
# print(probs.shape,corrects.shape)
#
# probs[np.where(corrects==1)]
#
# probs[corrects==1]
#
# probs.sum(axis=0)
#
# probs[range(probs.shape[1]),corrects]
#
# correct_logprobs = -np.log(probs*corrects)

print("Softmax-crossentropy loss: ",softmax_crossentropy_loss(data,corrects))
print("hinge loss:",hinge_loss(data,corrects,1))
