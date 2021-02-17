import tensorflow as tf
import numpy as np
import komm

'''
creating graph
creating a graph that has an 4D tensor as input and modulates the tensor and sends it in the AWGN channel. The output has the same shape as the input but may differ in respect to the values since the AWGN corupts the data

Note:the size of the input must be an even integer
'''
#creating the placeholder
Input=tf.placeholder(tf.float32, (None,None,None,None))

#reshape to shape: (N,2):

dim=tf.shape(Input)
target_dim=tf.convert_to_tensor(((dim[0]*dim[1]*dim[2]*dim[3])/2,2),dtype=tf.int32)
tensor2dim=tf.reshape(Input,target_dim)
print(np.shape(tensor2dim))

#convert from vector of shape (N,2) into a complex number of shape(N):

complexTensor=tf.dtypes.complex(tensor2dim[:,0],tensor2dim[:,1])
print(tf.shape(complexTensor))


#sending the array
def modulatedinchannel(modulated):
    awgn= komm.AWGNChannel(snr=100.0, signal_power=1.0)
    sentsignal=awgn(modulated)
    return sentsignal[0]#einfach ganz lollig
#map_fn requires a scalar output of the function as given by sentsignal[0]

inchannel=tf.map_fn(modulatedinchannel,complexTensor)




# complex shape(N,) to vector of shape (N,2)
out2dim=tf.transpose((tf.math.real(inchannel),tf.math.imag(inchannel)))
print(np.shape(out2dim))


#transform to the original array

originalshape=tf.reshape(out2dim,dim)

#for example
data=np.random.randint(0,100000,(24,28,28,7))


#unit test
ops=[Input,dim,tensor2dim,complexTensor,inchannel,out2dim,originalshape]
with tf.Session() as sess:

    n=0
    for op in ops:
        feed = {Input: data}
        output = sess.run(op, feed_dict=feed)
        n+=1
        print("\n#########################\nprocess:" +str(n))
        print(np.shape(output))


print(output==Input)

print("session finished")

