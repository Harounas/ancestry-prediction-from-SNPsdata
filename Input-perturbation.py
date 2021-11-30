from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras import optimizers
import pickle
import math
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils.data_utils import Sequence
from keras.models import Model
from keras.regularizers import l2,l1
import keras.backend as K
from keras.models import load_model, model_from_json
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Flatten, Dropout
from keras.constraints import unit_norm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import Normalizer
import tensorflow as tf
from tensorflow.contrib.data import Dataset
import itertools
from sklearn.metrics import confusion_matrix
sess = K.get_session()
with open("/home/haroun/1000genometraindata-svd.pickle", "rb") as g:
     X_train, y_train=pickle.load(g)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)

with open("/home/haroun/1000genometestdata-svd.pickle", "rb") as f:
     X_test,y_test=pickle.load(f)

X_test=scaler.fit_transform(X_test)
facto=0.0
noise = facto*np.random.normal(loc=0.0, scale=1, size=X_train.shape)
X_train_noisy = X_train + noise.astype('float32')

noise = facto*np.random.normal(loc=0.0, scale=1, size=X_test.shape)
X_test_noisy = X_test + noise
X_train_noisy, X_valid_noisy, y_train_noisy, y_valid_noisy = train_test_split(X_train_noisy, y_train, test_size=0.2, random_state=None)

class Generator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x_set, y_set, batch_size=4):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        return batch_x, batch_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
batch_size=32
epochs=3000
train_datagen = Generator(X_train_noisy, y_train_noisy, batch_size)#begin
test_datagen = Generator(X_valid_noisy, y_valid_noisy, batch_size)
#input_sample = Input(shape=(X_train.shape[1],))
#startds
input_sample =tf.placeholder(tf.float32,shape=(None, X_train.shape[1]))
output_sample =tf.placeholder(tf.float32,shape=(None, y_train.shape[1]))
inputs = Input(tensor=input_sample)

with open("classifier1000genome.json", "r") as f:
     model=model_from_json(f.read())
#load weights0
model.load_weights("classifier1000genome.h5")

model.layers[-1].name='dense_final'
mid_start = model.get_layer('dense_final')
all_layers = model.layers
for i in range(model.layers.index(mid_start)):
    all_layers[1].trainable =False
    all_layers[i+1].trainable =False
layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_name = 'dense_final'
print model.summary()

sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
adam=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping_monitor = EarlyStopping(patience=100)

with open('1000genometraindata.pickle', 'rb') as f:
   X, Y= pickle.load(f)
with open('1000genometestdata.pickle', 'rb') as f:
   X_test,y_test= pickle.load(f)
epsilon=0.01
x_adv= X_train
# Added noise
#prev_probs = []
loss=  -K.categorical_crossentropy(y_train, model.output)
grads = K.gradients(loss, model.input)
delta = K.sign(grads[0])
for i in range(5):
    x_adv = x_adv + epsilon*delta

x, _, _, _= np.linalg.lstsq(X, x_adv)
#print (x.shape)

X_new= np.dot(X,x)
#valid= np.dot(X_valid,x)
test=np.dot(X_test,x)
#print X_new.shape
#print test.shape

with open("/home/haroun/1000genomtrain1000p00.pickle", "wb") as f:
    pickle.dump((X_new,Y), f)
with open("/home/haroun/1000genometest1000p00.pickle", "wb") as f:
    pickle.dump((test, y_test), f)
#score = model.evaluate(test1 , y_test, batch_size=None, verbose=0, steps=1)
e=model.evaluate(test, y_test, batch_size=None, verbose=0, steps=1)
#print('Test loss:', e[0])
print('Test loss:', e[1], e[0])

