import tensorflow as tf
import numpy as np
from keras import backend as K
import komm
import keras as ks
from keras.layers import Conv2D, Input, Conv2DTranspose, Dense
from numpy.core._multiarray_umath import dtype
import matplotlib.pyplot as plt
import layer
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import savetraining


def processtrainingdata():
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(0,1)

    global x_train,x_test
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    print(x_train.shape)
    print(x_test.shape)

processtrainingdata()

#preparation to store critical info
path=savetraining.createfolder()

'''
########################################################################################################################
building the Graph

'''

input_dim=(28,28,1)




input_img_batch=tf.placeholder(dtype=float,shape=input_dim)
input_img_batch=ks.Input(shape=input_dim)
def encoder(input_img_batch):
    encoded = Conv2D(14, (6, 6), strides=(4, 4))(input_img_batch)
    encoded = Conv2D(14, (3, 3), strides=(2, 2))(encoded)
    return encoded

outofchannel=tf.placeholder(dtype=float,shape=(None,None,None,None))
def decoder(output_channel):
    output_img = Conv2DTranspose(14, (5, 5), strides=(2, 2))(output_channel)
    output_img = Conv2DTranspose(1, (4, 4), strides=(4, 4))(output_img)
    output_img = tf.math.abs(output_img)
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
loss=tf.reduce_sum(tf.square(output_img/255-input_img_batch/255))
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
        print("\n#########################################\nprocess " + str(n)+": "+ str(test_labels[n-1]))

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

print("########################################\nall tests successful\n########################################\n########################################\n\n\n")




epochs=3000
sampleimageindex=4
batchsize=300
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
    plt.figure(dpi=1200)

    for epoch in range(epochs):
        print("epoch: "+str(epoch+1))

        feed = {input_img_batch: x_train[np.random.randint(0, 59999, 50)]}#generating a random sample of imgs for batch
        epochimg,lossinepoch,traininginepoch = sess.run([output_img,loss,training], feed_dict=feed)

        #recording loss
        lossfn.append(lossinepoch)
        #representative image
        rep_img=np.reshape(epochimg[sampleimageindex],(28,28))
        subplt=plt.subplot(int(np.sqrt(epochs))+1,int(np.sqrt(epochs))+1,epoch+1,)
        subplt.get_xaxis().set_visible(False)
        subplt.get_yaxis().set_visible(False)
        print(str(lossinepoch))


        plt.imshow(rep_img )



    print("training finished")
    savetraining.saveplot(path, "sampleimgs")
    plt.show()  # sample img
    if (is_keras):
        savetraining.savenetwork(path,autoencoder,'testnetwork')

    plt.plot(range(len(lossfn)), lossfn, label='loss')
    plt.xlabel('epochs')
    plt.legend()
    savetraining.saveplot(path, "loss")
    plt.show()


    for imgset in range(1,10):

        indexes=range(imgset*10,imgset*10+5)
        num_index=len(indexes)
        imgs_opt=x_train[indexes]
        feed2={input_img_batch: imgs_opt}
        imgs_pred = sess.run([output_img], feed_dict=feed2)
        print(str(np.shape(imgs_pred)))
        print(str(np.shape(imgs_opt)))
        imgs_pred=np.reshape(imgs_pred,(num_index,28,28))
        imgs_opt = np.reshape(imgs_opt, (num_index, 28, 28))

        plt.figure()
        for index,place in zip(indexes,range(num_index)):
            img_pred=np.reshape(imgs_pred[place],(28,28))
            img_opt = np.reshape(imgs_opt[place], (28, 28))

            subplt1=plt.subplot(2,num_index,place+1)
            plt.imshow(img_pred,cmap='binary')
            plt.title("prediction")
            subplt1.get_xaxis().set_visible(False)
            subplt1.get_yaxis().set_visible(False)

            subplt2 = plt.subplot(2, num_index, place+num_index+1)
            plt.imshow(img_opt,cmap='binary')
            plt.title("original")
            subplt2.get_xaxis().set_visible(False)
            subplt2.get_yaxis().set_visible(False)

        savetraining.saveplot(path, "prediction_comparison"+str(imgset))
        plt.show()  # sample img












