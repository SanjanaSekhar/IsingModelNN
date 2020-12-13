#============================
# Train neural networks on NxN ising model configs
# Binary classification to predict if T>Tc
# Author: Sanjana Sekhar
# Date: 11 Dec 20
#============================

import h5py
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.callbacks import EarlyStopping


h5_date = "dec12"
img_ext = "dec13"
Tc = 2.268
accuracy_perT_allN = np.zeros((4,40))
j = 0

for N in [10,20,30,40]:

  # Load data
  f = h5py.File('h5_files/train_N%i_%s.hdf5'%(N,h5_date), 'r')
  ising_train = f['ising'][...]
  label_train = f['label'][...]
  mag_train = f['mag'][...]
  temp_train = f['temp'][...] 
  f.close()

  f = h5py.File('h5_files/test_N%i_%s.hdf5'%(N,h5_date), 'r')
  ising_test = f['ising'][...]
  label_test = f['label'][...]
  mag_test = f['mag'][...]
  temp_test = f['temp'][...]
  f.close()


  # Model configuration according to paper
  batch_size = 32
  loss_function = 'binary_crossentropy'
  n_epochs = 20
  optimizer = Adam()
  validation_split = 0.2

  train_time_s = time.clock()


  inputs = Input(shape=(N,N))
  x = Flatten()(inputs)
  x = Dense(100, kernel_regularizer='l2')(x)
  x = Activation("sigmoid")(x)
  x = Dense(1)(x)
  label = Activation("sigmoid", name="x")(x)



  model = Model(inputs=inputs,
                outputs=label
                )

   # Display a model summary
  model.summary()

  #history = model.load_weights("checkpoints/cp_%i_%s.ckpt"%(N,img_ext))

  # Compile the model
  model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy']
                )

  callbacks = [
  EarlyStopping(patience=2),
  ModelCheckpoint(filepath="checkpoints/cp_%i_%s.ckpt"%(N,img_ext),
                  save_weights_only=True,
                  monitor='val_loss')
  ]

  # Fit data to model
  history = model.fit(ising_train, label_train,
                  batch_size=batch_size,
                  epochs=n_epochs,
                  callbacks=callbacks,
                  validation_split=validation_split)

  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss of the network for N = %i'%(N))
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['training loss', 'validation loss'], loc='upper right')
  plt.savefig("plots/loss_N%i_%s.png"%(N,img_ext))
  plt.close()

  print("training time ",time.clock()-train_time_s)

  #Make a prediction on the test set

  label_pred = model.predict(ising_test, batch_size=64)
  loss_vals, accuracy = model.evaluate(ising_test,label_test)

  print('accuracy = ',accuracy)

  #plot avg output layer prediction per T
  #plot avg accuracy per T 

  T_list = np.linspace(1, 3.5, 40)
  output_per_T = np.zeros((40,1))
  accuracy_per_T = np.zeros((40,1))

  for index in range(len(T_list)):
    idx = np.argwhere(temp_test==T_list[index])[:,0]
    output_per_T[index]=np.mean(label_pred[idx])
    loss_vals, accuracy_per_T[index] = model.evaluate(ising_test[idx],label_test[idx])

  accuracy_perT_allN[j] = accuracy_per_T.transpose()
  j+=1

  
  plt.plot(T_list,output_per_T,'o-',label='T<Tc')
  plt.plot(T_list,(1-output_per_T),'o-',label='T>Tc')
  plt.plot([Tc,Tc],[0,1],label='Tc = 2.268')
  plt.xlabel('T/J')
  plt.ylabel('Output layer')
  plt.title('Output layer predictions for N = %i'%(N))
  plt.legend()
  plt.savefig("plots/output_per_T_N%i_%s"%(N,img_ext))
  plt.close()

  plt.plot(T_list,accuracy_per_T,'o-')
  plt.plot([Tc,Tc],[0,1],label='Tc = 2.268')
  plt.xlabel('T/J')
  plt.ylabel('Accuracy')
  plt.title('Accuracy per T for N = %i'%(N))
  plt.legend()
  plt.savefig("plots/accuracy_per_T_N%i_%s"%(N,img_ext))
  plt.close()

for index in [0,1,2,3]:
  plt.plot(T_list,accuracy_perT_allN[index],'o-',label='N = %i'%(10*(index+1)))

plt.plot([Tc,Tc],[0,1],label='Tc = 2.268')
plt.xlabel('T/J')
plt.ylabel('Accuracy')
plt.title('Accuracy per T')
plt.legend()
plt.savefig("plots/accuracy_per_T_allN_%s"%(img_ext))
plt.close()



