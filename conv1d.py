
import pandas as pd
import os

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
class SaveLastModelCallback(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        self.model.save('last_model.h5')
class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, df, batch_size=32,shuffle=True,image_dir=""):
        self.batch_size = batch_size
        self.df = df
        self.image_dir = image_dir
        self.indices = self.df.index.tolist()
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)
    
    

    def load_embedding(self, slice):
        data = np.load(self.df.iloc[slice]['data'], allow_pickle=True)
        embedding = data[0]
        label = data[1]
        return embedding, label

    def __get_data(self, batch):
        X = []
        y = []
        for i, id in enumerate(batch):
            img, label = self.load_embedding(batch[i])
       
            X.append(img)
            y.append(label)
        return np.array(X),np.array(y)

df_validation = pd.DataFrame({'data':[f"test_car/embeddings/{x}" for x in os.listdir('test_car/embeddings/')]})



df = pd.DataFrame({'data':[f"data/embeddings/{x}" for x in os.listdir('data/embeddings/')]})



df.head()



df_validation.head()


train_dataloader = DataGenerator(df)
val_dataloader = DataGenerator(df_validation)


import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D,Dropout
from numpy import unique



from keras import backend as K

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense,BatchNormalization

model = Sequential()
model.add(Conv1D(filters=256, kernel_size=5, activation='relu', padding='same', input_shape=(1536, 1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer='l2', bias_regularizer='l2'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[precision_m,'accuracy'])

if os.path.exists('no_conv_last_model.h5'):
    model.load_weights('no_conv_last_model.h5')
elif os.path.exists('no_conv.h5'):
    model.load_weights('no_conv.h5')




es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
mcp = tf.keras.callbacks.ModelCheckpoint(
    filepath='no_conv.h5',
    save_weights_only=True,
    monitor='val_precision_m',
    mode='max',
    save_best_only=True)
import datetime
log_dir = f"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
save_last_model = SaveLastModelCallback()


model.fit(train_dataloader, validation_data=val_dataloader, epochs = 100, batch_size = 128, callbacks=[mcp,tensorboard_callback,es,save_last_model])
model.save('last.h5')

