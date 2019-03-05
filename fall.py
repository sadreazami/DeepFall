
import scipy.io as sio
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import math
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  
from sklearn import svm
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import cross_val_score  
import scipy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.image as mpimg
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import os
from scipy import ndimage, misc
path = 'C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/imagenaug/'
#path = 'C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/Dataset and Labels/new Dataset and Lables/imagenew2/'
SSS=2060
data = np.zeros((SSS,64,69))
zz=[]

for i in range(SSS):    
    zz.append(str(i)+".bmp")

for ii, imagee in enumerate(zz):
    path2 = os.path.join(path, imagee)
    image2 = ndimage.imread(path2)
    image2=image2.astype(np.float64)
    image2= scipy.misc.imresize(image2, 0.5)
#    data2 = img_to_array(image2)
#    data2=np.squeeze(data2)
#    data = image3.reshape(129*139, 1)
    data[ii,:,:]=image2/255

import csv
with open('C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/Final/lab.csv', 'r') as mf:
#with open('C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/Dataset and Labels/new Dataset and Lables/lnew.csv', 'r') as mf:
     re = csv.reader(mf,delimiter=',',quotechar='|')
     re=np.array(list(re))
     label = re.astype(np.float64)
     label=np.squeeze(label)  
    
label=np.repeat(label,10)
#label=np.hstack((label,1))
########################## Fully connected layers information #################################################

FullyConnected_active = True

                                ## number of input layer units (features)
#M=[1000,300,200,100,50,10,2]     ## Hidden layers units for fully_connected layers
#M=[2] 
#M=[500,200,100,50,2]  
M=[500, 2]        
############################################################################################################

def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
             return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
    else:
             return tf.nn.batch_normalization(inputs,pop_mean, pop_var, beta, scale, epsilon)              

######################### Convolutional layer information #####################################################

convolutional_active = False

################### for conv2D
epsilon = 1e-3
conv_strides = [[1,1,1,1],[1,1,1,1]]
filter_size = [[3,3,1,64],[3,3,64,128]]

################### for pooling

ksize = [[1,2,2,1],[1,2,2,1]]
pool_strides = [[1,2,2,1],[1,2,2,1]]

################################################################################################################

tf.reset_default_graph()

m=4

kf=KFold(5, random_state=False, shuffle=False)
kf.get_n_splits(data)
k=0
for train_index, test_index in kf.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    if k==m:
       break 
    k=k+1

#n_samples = 64*69  
#X_train=data[:1580,:,:]
#X_test=data[1580:,:,:]

#X_train = X_train.reshape([158,n_samples, -1])
#X_test = X_test.reshape([48,n_samples, -1])
#X_train=np.squeeze(X_train) 
#X_test=np.squeeze(X_test) 
    
#y_train = label[:1580]
#y_test = label[1580:]
       
#########################################################################################################

X = tf.placeholder(dtype=tf.float32 , shape=[None,64,69] , name='input')
Y = tf.placeholder(dtype=tf.int64 , shape = [None,], name='output')

phase_train = tf.placeholder(tf.bool, name='phase_train')

p = tf.placeholder(dtype=tf.float32, name = 'Drop_out')

###########################################################################################################################
################################## Convolutional layers design  ###########################################################
###########################################################################################################################
Frame= tf.expand_dims(X,3)

if convolutional_active == True :     # if convolutional is used

        
   for index, (ks, Pstrd, Cstrd , fil) in enumerate(zip(ksize,pool_strides,conv_strides,filter_size)):

        with tf.variable_scope ('conv'+str(index)) as scope:
             convW= tf.get_variable('convW', fil, initializer = tf.truncated_normal_initializer(stddev=0.01))
             fcb= tf.get_variable('bias', initializer =tf.constant(0.01,shape=[fil[-1]]))
             #A  = tf.gradient(convW)

             ConV=tf.nn.conv2d(Frame,convW, strides = Cstrd,  padding='SAME', name='conv')
             ConV=batch_norm_wrapper(ConV,True)
             Relu=tf.nn.relu(ConV + fcb , name='relu')
             pool = tf.nn.max_pool(Relu, ksize=ks , strides = Pstrd , padding='SAME', name = 'pooling')
             Frame=pool
                               
###########################################################################################################################
################################### Fully _ connected layers design ##########################################
##############################################################################################################
size = Frame.get_shape().as_list()
N=size[1]*size[2]*size[3]
inpt=tf.reshape(Frame,[-1, N])

if FullyConnected_active == True : 
       
   for m in range(len(M)):        
       with tf.variable_scope ('FullyC'+str(m)) as scope:
             fcW= tf.get_variable('fcW', [N,M[m]] , initializer = tf.truncated_normal_initializer(stddev=1))* tf.sqrt(2/N)
             fcB= tf.get_variable('bias', initializer=tf.constant(0.001,shape=[M[m]])) 
             Z=tf.add(tf.matmul(inpt,fcW) , fcB, name= 'add') 
             
             if m < len(M)-1:
                                                    
                inpt= tf.nn.dropout(tf.nn.relu(Z), p)                  
                N=M[m] 
             else:
                h=tf.nn.softmax(Z)                                
###################################################################################################################
vars   = tf.trainable_variables() 
lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'fcb' not in v.name ]) 

loss= tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=Z)) #+ 0.00001* lossL2
 
var_list=tf.trainable_variables() 
optimizer=tf.train.AdamOptimizer(0.01).minimize(loss)
    
saver=tf.train.Saver()
init=tf.global_variables_initializer()

correct_p= tf.equal(tf.argmax(h,1), Y)

s1=tf.cast(tf.equal(Y,1), tf.float32)
s0=tf.cast(tf.equal(Y,0), tf.float32)

s_1=tf.reduce_sum(s1)   #  label 1
s_0=tf.reduce_sum(s0)   #  label 0

c1=tf.cast(tf.equal(tf.argmax(h,1),1) , tf.float32)
c0=tf.cast(tf.equal(tf.argmax(h,1),0) , tf.float32)

c_1 = tf.reduce_sum(c1)  #
c_0 = tf.reduce_sum(c0)  #

c_1_S_1 = tf.reduce_sum(tf.cast((tf.equal(c1,1) & tf.equal(s1,1)),tf.float32))  #
c_1_S_0 = tf.reduce_sum(tf.cast((tf.equal(c1,1) & tf.equal(s0,1)),tf.float32))   #
c_0_S_0 = tf.reduce_sum(tf.cast((tf.equal(c0,1) &tf.equal(s0,1)),tf.float32))
c_0_S_1 = tf.reduce_sum(tf.cast((tf.equal(c0,1) & tf.equal(s1,1)),tf.float32))   #


Acuraccy = tf.reduce_mean(tf.cast(correct_p, tf.float32))

###################################################################################################################
saver = tf.train.Saver()

with tf.Session() as sess:
     sess.run(init)
    
     for ep in range (200):
             
         _,L = sess.run([optimizer,loss], feed_dict={ X:X_train , Y:y_train , p : 0.8 }) 

         A = sess.run(Acuraccy, feed_dict={ X:X_train , Y: y_train, p : 1 })
             
         At, c1S1, c0S0, c1S0, c0S1 = sess.run([Acuraccy , c_1_S_1, c_0_S_0, c_1_S_0, c_0_S_1 ], feed_dict={ X:X_test , Y: y_test, p : 1 })

         print(ep,"  ",i, "  L=  ", L , "acc = ",A , "Acc = ", At,  c1S1," ",c0S0, " ",c1S0,"  ",c0S1)

