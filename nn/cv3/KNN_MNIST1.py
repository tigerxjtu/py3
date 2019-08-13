from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt

# importation of Mnist data
mnist = datasets.load_digits()
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),mnist.target,test_size= 0.25, random_state= 42)
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData,trainLabels,test_size= 0.1, random_state= 84)

# Make model and prediction
k = 5
model = KNeighborsClassifier(n_neighbors= k,algorithm='kd_tree')
model.fit(trainData, trainLabels)
prediction = model.predict(testData)
score = model.score(valData,valLabels)



print("accuracy = %.2f%%" %(score * 100))

# plot the test and result
image = testData
j = 0
for i in np.random.randint(0,high=len(testLabels),size=(24,)):
    prediction = model.predict(image)[i]
    image0 = image[i].reshape((8,8))
    plt.subplot(4,6,j+1)
    plt.title(str(prediction))
    plt.imshow(image0,cmap='gray')
    plt.axis('off')
    j = j+1

plt.show()