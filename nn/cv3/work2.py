from sklearn.neighbors import KNeighborsClassifier
import numpy as np


data = np.array([[1, 1, 1, 1],
              [0.5, 1, 1, 1],
              [0.1, 0.1, 0.1, 0.1],
              [0.5, 0.5, 0.5, 0.5],
              [1, 0.8, 0.3, 1],
              [0.6, 0.5, 0.7, 0.5],
              [1, 1, 0.9, 0.5],
              [1, 0.6, 0.5, 0.8],
              [0.5, 0.5, 1, 1],
              [0.9, 1, 1, 1],
              [0.6, 0.6, 1, 0.1],
              [1, 0.8, 0.5, 0.5],
              [1, 0.1, 0.1, 1],
              [1, 1, 0.7, 0.3],
              [0.2, 0.3, 0.4, 0.5],
              [0.5, 1, 0.6, 0.6]
              ])

labels = ['美女',
          '淑女',
          '丑女',
          '一般型',
          '淑女',
          '一般型',
          '美女',
          '一般型',
          '淑女',
          '美女',
          '丑女',
          '可爱型',
          '可爱型',
          '淑女',
          '丑女',
          '可爱型'
          ]

k = 5
model = KNeighborsClassifier(n_neighbors= k,algorithm='kd_tree')
model.fit(data,labels)

record=[0.5, 1, 1, 1]
prediction = model.predict(np.reshape(record,(1,-1)))
print(prediction)
