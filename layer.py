import tensorflow as tf
from keras import backend as K
import numpy as np
import komm
import keras as ks


print("executing_layer")

'''
takes a tensor of shape(None,None,None,None) and simulates the the process of sending and receiving the tensor by using and AWGN channel
'''
def AWGNlayer(tensor4dim):
    from keras import backend

    #creating the placeholder
    Input=tensor4dim
    #reshape to shape: (N,2):

    dim=tf.shape(Input)
    target_dim=tf.convert_to_tensor(((dim[0]*dim[1]*dim[2]*dim[3])/2,2),dtype=tf.int32)
    tensor2dim=tf.reshape(Input,target_dim)
    complexTensor=tf.dtypes.complex(tensor2dim[:,0],tensor2dim[:,1])


    #convert from vector of shape (N,2) into a complex number of shape(N):



    def modulatedinchannel(modulated):
        awgn= komm.AWGNChannel(snr=100.0, signal_power=1.0)
        sentsignal=awgn(modulated)
        return sentsignal[0]#einfach ganz lollig
    #map_fn requires a scalar output of the function as given by sentsignal[0]

    inchannel=tf.map_fn(modulatedinchannel,complexTensor)




    #sending the array

    # complex shape(N,) to vector of shape (N,2)
    out2dim=tf.transpose((tf.math.real(inchannel),tf.math.imag(inchannel)))



    #transform to the original array

    originalshape=tf.reshape(out2dim,dim)

    return originalshape

def AWGNshapes(tensor_shape):
    from keras import backend

    return tensor_shape

AWGNlayer=ks.layers.Lambda(AWGNlayer,AWGNshapes)



'''
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

'''

print("layer successful executed")