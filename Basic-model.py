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
import keras
from keras.models import model_from_json
from keras.constraints import unit_norm
from keras.layers import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Flatten, Dropout
from keras import regularizers
from keras.constraints import max_norm
from keras import losses
from numpy import mean
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from keras.constraints import maxnorm
from keras.layers import Input, Dense, Lambda
import tensorflow as tf
sess = tf.Session()
K.set_session(sess)
with open("/home/haroun/1000genomtraindata-svd.pickle", "rb") as g:     
     X_train, y_train=pickle.load(g)

scaler = StandardScaler()

X_train=scaler.fit_transform(X_train)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=None)
print X_train.shape
with open("/home/haroun/1000genometestdata-svd.pickle", "rb") as f:
     X_test,y_test=pickle.load(f)
X_test=scaler.fit_transform(X_test)
facto=0.0
noise = facto*np.random.normal(loc=0, scale=0.5, size=X_train.shape)
X_train_noisy = X_train + noise
noise = facto*np.random.normal(loc=0, scale=0.5, size=X_valid.shape)
X_valid_noisy = X_valid + noise
noise = facto*np.random.normal(loc=0, scale=0.5, size=X_test.shape)
X_test_noisy = X_test + noise
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
a=0.0001
def contractive_loss(y_true, y_pred):
        cat= losses.categorical_crossentropy(y_true, y_pred)
        h = classifier.get_layer('encoded1').output
        W = K.variable(value=classifier.get_layer('encoded1').get_weights()[0])
        W = K.transpose(W)
        return cat + a*K.sqrt(K.sum(h**2, axis=1))
batch_size=32
epochs=2000
train_datagen = Generator(X_train_noisy,y_train, batch_size)

test_datagen = Generator(X_valid_noisy, y_valid, batch_size)
input_sample = Input(shape=(X_train.shape[1],))
inputs = Input(tensor=input_sample)
inputs = Lambda(lambda x: tf.cast(x, tf.float32))(inputs)
encoder = Dense(units=100, activation='linear', name='encoded1')(input_sample)
encoder = BatchNormalization()(encoder)
encoder = Dropout(0.5)(encoder)
encoder = Dense(units=100, activation='relu')(encoder)
encoder = BatchNormalization()(encoder)
encoder = Dropout(0.5)(encoder)
output = Dense(26,activation='softmax', name='main_output')(encoder)
classifier = Model(inputs=input_sample,outputs=output)
classifier.summary()
adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-4, amsgrad=False)
sgd = optimizers.SGD(lr=0.001, decay=1e-8, momentum=0.9, nesterov=True)
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
early_stopping_monitor = EarlyStopping(patience=100)
history=classifier.fit_generator(train_datagen,
	steps_per_epoch=len(X_train)//batch_size,
	validation_data=test_datagen,
	validation_steps=len(X_valid)//batch_size,
	epochs=epochs, callbacks=[early_stopping_monitor])
score = classifier.evaluate(X_test_noisy, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
classifier.save_weights('classifier_weghts1000genome.h5')

with open('classifier1000genome.json', 'w') as f:
    f.write(classifier.to_json())

plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



