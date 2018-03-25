import numpy as np
import tensorflow as tf
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import sigmoid
from keras.losses import MSE
from keras.optimizers import SGD
from keras.metrics import binary_accuracy
sess=tf.Session()
k.set_session(sess)

logicand=np.array([[0,0,0],
                 [0,1,0],
                 [1,0,0],
                 [1,1,1]])


logicor=np.array([[0,0,0],
                 [0,1,1],
                 [1,0,1],
                 [1,1,1]])

logicxor=np.array([[0,0,0],
                 [0,1,1],
                 [1,0,1],
                 [1,1,0]])

logicnot=np.array([[0,1],
                 [1,0]])
# x=logicand[:,:2]
# y=logicand[:,-1]

# x=logicor[:,:2]
# y=logicor[:,-1]

x=logicxor[:,:2]
y=logicxor[:,-1]

# x=logicnot[:,:1]
# y=logicnot[:,-1]



model = Sequential()
model.add(Dense(2,activation=sigmoid,input_dim=2))
model.add(Dense(1,activation=sigmoid))
model.compile(loss=MSE,optimizer=SGD(lr=1))
model.fit(x,y,epochs=1000)
model.save("xorGates.h5")
#print(model.predict(np.array([[,1]])))x

# model = Sequential()
# model.add(Dense(1,activation=sigmoid,input_dim=1))
# model.compile(loss=MSE,optimizer=SGD(lr=1))
# model.fit(x,y,epochs=10000)
# model.save("notGAtes.h5")
