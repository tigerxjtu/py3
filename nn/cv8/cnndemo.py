
import numpy as np

def convolve(src,kernel):
    # result=np.dot(src,kernel)
    result=src*kernel
    # print(src,kernel,np.sum(result))
    return np.sum(result)

def max_pooling(src,pool):
    w,h=pool
    width,height=int(src.shape[0]/w),int(src.shape[1]/h)
    result=np.zeros((width,height))
    for i in range(width):
        for j in range(height):
            result[i,j]=np.max(src[i*w:(i+1)*w,j*h:(j+1)*h])
    return result

def conv2d(input, filter, stride):
    x_size,y_size=input.shape[0],input.shape[1]
    k_x,k_y=filter.shape[0],filter.shape[1]
    w,h=int((x_size-k_x)/stride+1),int((y_size-k_y)/stride+1)
    result=np.zeros((w,h))
    for i in range(0,x_size,stride):
        if i+k_x>x_size:
            continue
        for j in range(0,y_size,stride):
            if j+k_y>y_size:
                continue
            row=int(i/stride)
            col=int(j/stride)
            # print(row,col,convolve(input[i:i+k_x,j:j+k_y],filter))
            result[row,col]=convolve(input[i:i+k_x,j:j+k_y],filter)
    return result


data_r=[[1,8,3,3,4],
        [4,5,4,6,2],
        [2,1,1,8,9],
        [6,10,9,8,2],
        [6,10,6,3,2]]

data_g=[[3,3,2,3,3],
        [10,2,5,6,10],
        [1,3,4,8,1],
        [6,3,5,6,6],
        [6,5,3,5,6]]

data_b=[[4,7,7,1,8],
        [1,8,10,9,8],
        [9,9,8,4,9],
        [8,6,6,6,9],
        [4,9,10,5,2]]

filter1_data=[3,4,3,4,8,3,8,7,8,
         5,7,5,3,6,7,3,8,2,
         9,9,7,5,2,2,8,7,10]

filter2_data=[3,10,6,2,1,8,8,9,5,
         4,2,5,1,10,6,4,9,10,
         3,6,10,1,6,2,3,2,4]

data=np.zeros((5,5,3))
data[:,:,0]=np.array(data_r)
data[:,:,1]=np.array(data_g)
data[:,:,2]=np.array(data_b)



filter1=np.zeros((3,3,3))
filter1[:,:,0]=np.array(filter1_data[:9]).reshape(3,3)
filter1[:,:,1]=np.array(filter1_data[9:18]).reshape(3,3)
filter1[:,:,2]=np.array(filter1_data[18:27]).reshape(3,3)

filter2=np.zeros((3,3,3))
filter2[:,:,0]=np.array(filter2_data[:9]).reshape(3,3)
filter2[:,:,1]=np.array(filter2_data[9:18]).reshape(3,3)
filter2[:,:,2]=np.array(filter2_data[18:27]).reshape(3,3)

# print(data.shape,data)
# print(filter1.shape,filter1)

# print(max_pooling(filter2[:,:,0]))
stride=2
print('----------con2d seperate---------')
r1=conv2d(np.array(data_r), np.array(filter1_data[:9]).reshape(3,3), stride)
r2=conv2d(np.array(data_g), np.array(filter1_data[9:18]).reshape(3,3), stride)
r3=conv2d(np.array(data_b), np.array(filter1_data[18:]).reshape(3,3), stride)

print(r1+r2+r3)

print('----------con2d together---------')
result1_conv=conv2d(data,filter1,stride)
result2_conv=conv2d(data,filter2,stride)

print(result1_conv)
print(result2_conv)

print('----------max_pooling-------------')
pool_shape=(2,2)
result1_max_pooling=max_pooling(result1_conv,pool_shape)
result2_max_pooling=max_pooling(result2_conv,pool_shape)


result=np.zeros((result1_max_pooling.shape[0],result1_max_pooling.shape[1],2))
result[:,:,0]=result1_max_pooling
result[:,:,1]=result2_max_pooling

print("result shape:",result.shape)
print(result)
