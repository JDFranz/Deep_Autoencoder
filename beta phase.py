import tensorflow as tf
import numpy as np
import komm
import keras as ks
from keras.layers import Conv2D, Input, Conv2DTranspose
from numpy.core._multiarray_umath import dtype
import matplotlib.pyplot as plt
import layer
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import savetraining

#preparation to store critical info
path=savetraining.createfolder()

def processtrainingdata():
    global x_train,x_test
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    print(x_train.shape)
    print(x_test.shape)

processtrainingdata()


'''
########################################################################################################################
building the Graph

'''

input_dim=(28,28,1)




input_img_batch=tf.placeholder(dtype=float,shape=input_dim)
input_img_batch=ks.Input(shape=input_dim)
def encoder(input_img_batch):
    encoded=Conv2D(14,(6,6),strides=(2,2))(input_img_batch)
    encoded = Conv2D(14, (3, 3), strides=(2, 2))(encoded)
    return encoded

outofchannel=tf.placeholder(dtype=float,shape=(None,None,None,None))
def decoder(output_channel):
    output_img=Conv2DTranspose(14,(4,4),strides=(2,2))(output_channel)
    output_img = Conv2DTranspose(1, (6, 6), strides=(2, 2))(output_img)
    output_img = output_img
    return output_img

#metagraph
encoded=encoder(input_img_batch)
output_channel=layer.AWGNlayer(encoded)
output_img=decoder(output_channel)

#creating models
is_keras=int(input("consists your network only of keras layers?[1/0]" ))
if (is_keras==1):
    autoencoder=ks.Model(input_img_batch,output_img)


#traingingparameters
loss=tf.reduce_sum(tf.square(output_img-input_img_batch))
optimizer= tf.train.AdamOptimizer(learning_rate=0.001)
training=optimizer.minimize(loss)
init = tf.global_variables_initializer()


feed={input_img_batch:x_train[np.random.randint(0,59999,50)]}

#testing
ops = [input_img_batch,encoded,output_channel,output_img]
test_labels=['test placeholder','test encoder','test AWGN channel','test decoder']
with tf.Session() as sess:
    sess.run(init)
    n = 0
    for op in ops:
        n += 1
        print("\n#########################\nprocess " + str(n)+": "+ str(test_labels[n-1]))

        Input_placeholder=sess.run(input_img_batch, feed_dict=feed)
        output = sess.run(op, feed_dict=feed)
        # schauen ob der feed passt
        print("\ninput Information: ")
        print("shape: " + str(Input_placeholder.shape))
        print("dtype: " + str(Input_placeholder.dtype))
        # schauen ob der feed passt

        print("\noutput Information: ")
        print("shape: " + str(output.shape))
        print("dtype: " + str(output.dtype))

#testing batches
print("\n\nbatch test result: ")
randindex=np.random.randint(0,59999,100)#generates random indexes
batch=x_train[randindex]
print(str(batch.shape)+str(batch.dtype))

print("##################################\nall tests successful\n##################################\n##################################\n\n\n")




epochs=500
sampleimageindex=4
batchsize=100
lossfn=[]
imgprogress=np.empty(1)

trainval=input("graph clear for training?\n for launch: 1\n to abort:0")

if (int(trainval)==1):
    pass
else:
    exit(0)

with tf.Session() as sess:
    print("\n########################################\ntraining:\n########################################")

    sess.run(init)
    plt.figure(figsize=(int(np.sqrt(epochs))+1, int(np.sqrt(epochs))+1))

    for epoch in range(epochs):
        print("epoch: "+str(epoch+1))

        feed = {input_img_batch: x_train[np.random.randint(0, 59999, 50)]}#generating a random sample of imgs for batch
        epochimg,lossinepoch,traininginepoch = sess.run([output_img,loss,training], feed_dict=feed)

        #recording loss
        lossfn.append(lossinepoch)
        #representative image
        rep_img=np.reshape(epochimg[sampleimageindex],(28,28))
        subplt=plt.subplot(int(epochs/10),int(epochs/10),epoch+1)
        subplt.get_xaxis().set_visible(False)
        subplt.get_yaxis().set_visible(False)


        plt.imshow(rep_img)

    print("training finished")
    savetraining.saveplot(path, "sampleimgs")
    plt.show()  # sample img
    if (is_keras):
        savetraining.savenetwork(path,autoencoder,'testnetwork')



plt.plot(range(len(lossfn)),lossfn,label='loss')
plt.xlabel('epochs')
plt.legend()
savetraining.saveplot(path,"loss")
plt.show()