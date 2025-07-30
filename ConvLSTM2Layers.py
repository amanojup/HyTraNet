# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 17:53:50 2022

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from keras.layers import Input, Conv3D,Conv2D, ConvLSTM2D, Conv2DTranspose, BatchNormalization, Flatten, Reshape, Dropout
from keras.layers import TimeDistributed, LSTM, Concatenate
from keras.models import Model
from keras import regularizers
import cv2
from keras.models import save_model, load_model
from keras import backend as K
from keras import backend as back
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
#from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from keras import metrics
import os
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
def accuracy(true,pred):
    return accuracy_score(true,pred)



data_path = r'D:\8_Article2021_ImageBased\ImageBasedPaper\GoogleImage\Wmask'
img_list=os.listdir(data_path)

#img_list[0]
img_list.sort()
len(img_list)
img_data_list=[]
test_holder=[]
for img in img_list:
    #input_img=cv2.imread(data_path+img)
    #input_img1 = cv2.imread(os.path.join(data_path, img))
    input_img = cv2.imread(os.path.join(data_path, img))
    test_img=cv2.imread(os.path.join(data_path, img),0)
    input_img=input_img.flatten()
    test_img=test_img.flatten()
    test_holder.append(test_img)
    img_data_list.append(input_img)
    ing = input_img.reshape(600, 1200,3)
    #cv2.imshow('o',ing)
    #cv2.waitKey(0)
    #%%
""" converting congestion class into categories"""
data3=np.array(test_holder)
test_holder = []
gray=np.zeros_like(data3)
gray[np.logical_and(data3>=0,data3<40)]=0
gray[np.logical_and(data3>=40,data3<140)]=1 
gray[np.logical_and(data3>=190,data3<255)]=2
gray[np.logical_and(data3>=140,data3<190)]=3
data3=pd.DataFrame(gray)
#%%
row = 600
col = 1200
channel = 3
chan = 1                    
data_per_day = 120
past_sequence = 1
prediction_horizon = 1
#split data into train and test based on number of day
train_upto_day = 10
total_day = 12
k = past_sequence + 1
""" TRAIN and VALIDATION x and y data"""
#%%
#y_train=[]
y_train=[]
b=[i for i in range(0,train_upto_day)]
for j in range (0,len(b)-1):
    train1=data3.iloc[b[j]*data_per_day : b[j+1]*data_per_day,::]
    for i in range(past_sequence+prediction_horizon,train1.shape[0]+1):
        d=train1.iloc[i-past_sequence:i,:]
        d=np.array(d)
        y_train.append(d)   
y_train=np.array(y_train)
train1=[]
y_train= y_train.reshape(y_train.shape[0],y_train.shape[1],row,col,chan)

y_vali=[]
b=[i for i in range(train_upto_day-1,total_day)]
for j in range (0,len(b)-1):
    vali1=data3.iloc[b[j]*data_per_day : b[j+1]*data_per_day,::]
    for i in range(past_sequence+prediction_horizon,vali1.shape[0]+1):
        d=vali1.iloc[i-past_sequence:i,:]
        d=np.array(d)
        y_vali.append(d)   
y_vali=np.array(y_vali)
vali1 =[]
y_vali = y_vali.reshape(y_vali.shape[0], y_vali.shape[1], row, col, chan)
y_train1=to_categorical(y_train)
print(y_train.shape)
y_vali1=to_categorical(y_vali)
print(y_vali.shape)
#%%
data1=np.array(img_data_list)
img_data_list =[]
data=pd.DataFrame(data1)
x_train=[]
b=[i for i in range(0,train_upto_day)]
for j in range (0,len(b)-1):
    train=data.iloc[b[j]*data_per_day:b[j+1]*data_per_day-prediction_horizon,::]
    for i in range(past_sequence,train.shape[0]+1):
        d=train.iloc[i-past_sequence:i,:]
        d=np.array(d) 
        x_train.append(d)
x_train=np.array(x_train)
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],row,col,channel)
print(x_train.shape)

x_vali=[]
b=[i for i in range(train_upto_day-1, total_day)]
b
for j in range (0,len(b)-1):
    vali = data.iloc[b[j]*data_per_day:b[j+1]*data_per_day-prediction_horizon,::]
    for i in range(past_sequence,vali.shape[0]+1):
        d = vali.iloc[i-past_sequence:i,:]
        d = np.array(d) 
        x_vali.append(d)
x_vali=np.array(x_vali)
x_vali=x_vali.reshape(x_vali.shape[0],x_vali.shape[1],row,col,channel)
print(x_vali.shape)
data = []
#%%
""" Architecture design""" 
step = 1
#row,col,channel= 600,1200,3
f1=4
f2=64
f3=96
f4=128
f5=160
f6=192
input_img = Input(shape=(step,row,col, channel))  
x1 = TimeDistributed(Conv2D(4, (3, 3),strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(input_img)
x1 = TimeDistributed(BatchNormalization())(x1)
x1 = TimeDistributed(Dropout(0.1))(x1)

x2 = TimeDistributed(Conv2D(4, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(x1)
x2 = TimeDistributed(BatchNormalization())(x2)
x2 = TimeDistributed(Dropout(0.1))(x2)
x2 = TimeDistributed(Conv2D(4, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(x2)
x2 = TimeDistributed(BatchNormalization())(x2)
x2 = TimeDistributed(Dropout(0.1))(x2)

x3 = TimeDistributed(Conv2D(4, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(x2)
x3 = TimeDistributed(BatchNormalization())(x3)
x3 = TimeDistributed(Dropout(0.1))(x3)
x3 = TimeDistributed(Conv2D(4, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(x3)
x3 = TimeDistributed(BatchNormalization())(x3)
x3 = TimeDistributed(Dropout(0.1))(x3)


k1 = ConvLSTM2D(4, (3,3),strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform',return_sequences=True,dropout=0.1)(x3)
k1 = BatchNormalization()(k1)
k2= ConvLSTM2D(4, (3,3),strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform',return_sequences=True,dropout=0.1)(k1)
k2 = BatchNormalization()(k2)

d4 = TimeDistributed(Conv2DTranspose(4, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(k2)
d4 = TimeDistributed(BatchNormalization())(d4)
d4 = Concatenate()([d4,x2])
d4 = TimeDistributed(Dropout(0.1))(d4)
d4 = TimeDistributed(Conv2D(4, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(d4)
d4 = TimeDistributed(BatchNormalization())(d4)
d4 = TimeDistributed(Dropout(0.1))(d4)

d5 = TimeDistributed(Conv2DTranspose(4, (2,2), strides=(2,2),activation='relu', padding='valid', kernel_initializer='he_uniform'))(d4)
d5 = TimeDistributed(BatchNormalization())(d5)
d5 = Concatenate()([d5,x1])
d5 = TimeDistributed(Dropout(0.1))(d5)
d5 = TimeDistributed(Conv2D(4, (3, 3), strides=(1,1),activation='relu', padding='same', kernel_initializer='he_uniform'))(d5)
d5 = TimeDistributed(BatchNormalization())(d5)
d5 = TimeDistributed(Dropout(0.1))(d5)

decoded = TimeDistributed(Conv2D(4, (3, 3), activation='softmax', padding='same',kernel_initializer='he_uniform'))(d5)

 
predNet = Model(input_img, decoded) 
predNet.summary()

#predNet = multi_gpu_model(predNet, gpus= 4)
predNet.compile(optimizer= 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
predNet.fit(x_train,y_train1, epochs=35, batch_size = 32, validation_data=(x_vali, y_vali1))
predNet.save('D:\8_Article2021_ImageBased/model1120.Convlstm')
#%%
#predNet1=load_model('D:\8_Article2021_ImageBased/model1120.h5')
#%%
""" TEST data x and Y"""
test_data_dir = r'D:\8_Article2021_ImageBased\ImageBasedPaper\GoogleImage\Wmask'
test_holder1 = []
test_list = os.listdir(test_data_dir)
test_list.sort()
for each in test_list:
    test = cv2.imread(os.path.join(test_data_dir, each))
    test = test.flatten()
    test_holder1.append(test)
#%%
data1=np.array(test_holder1)
img_data_list =[]
data=pd.DataFrame(data1)
    
x_test=[]
b=[i for i in range(12,15)]
for j in range (0,len(b)-1):
    test = data.iloc[b[j]*data_per_day:b[j+1]*data_per_day-prediction_horizon,::]
    for i in range(past_sequence,test.shape[0]+1):
        d=test.iloc[i-past_sequence:i,:]
        d=np.array(d) 
        x_test.append(d)
x_test=np.array(x_test)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],row,col,channel)
print(x_test.shape)
#%%
xx = predNet.predict(x_test, batch_size = 32)
predict = xx[:,:,:,:]
predict = predict.argmax(axis = -1)
predict1 = predict.reshape(len(x_test), 600*1200)
#predict = predict.reshape(y_vali.shape[0],1, row, col, 1)
predict_n=np.zeros(shape=(predict.shape[0],1,600,1200,3))
predict_n[np.logical_and(predict>=0,predict<1)]=[0,0,0]
predict_n[np.logical_and(predict>=1,predict<2)]=[75,90,255]
predict_n[np.logical_and(predict>=2,predict<3)]=[75,225,250]
predict_n[np.logical_and(predict>=3,predict<=4)]= [100,195,140]
print(predict_n.shape)
predict_n = predict_n.reshape(x_test.shape[0], 600, 1200, 3)
#predict_n = predict_n.argmax(axis = 0)
#gray_image = cv2.cvtColor(predict_n[0], cv2.COLOR_GRAY2RGB
cv2.imwrite('D:\8_Article2021_ImageBased\ImageBasedPaper\GoogleImage\i1.png', predict_n[0])
cv2.imshow('i', predict_n[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
y_test=[]
b=[i for i in range(12,15)]
for j in range (0,len(b)-1):
    test1=data3.iloc[b[j]*data_per_day : b[j+1]*data_per_day,::]
    for i in range(past_sequence+prediction_horizon,test1.shape[0]+1):
        d=test1.iloc[i-past_sequence:i,:]
        d=np.array(d)
        y_test.append(d)   
y_test=np.array(y_test)
test1 =[]
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1],row, col,1)
y_test1=to_categorical(y_test)

y_test1 = y_test1.argmax(axis= -1)
y_test1 = y_test1.reshape(len(x_test), 600*1200)
#%%
MSE=mean_squared_error(y_test1,predict1)
MAE=mean_absolute_error(y_test1,predict1)
#h= []
for i in range(0,x_vali.shape[0]):
    x = accuracy_score(y_test1[i],predict1[i])
   #h.append(x)
   
print(MSE)
print(MAE)
print(x)