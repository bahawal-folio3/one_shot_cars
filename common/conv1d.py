import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

parser = argparse.ArgumentParser(description="This is a simple argument parser template.")
parser.add_argument("train_embedding_source", help="Path to the folder containing embeddings")
parser.add_argument("val_embedding_source", help="Path to the folder containing validation embeddings")
parser.add_argument("model_path_save", help="Path where generated embedding will be saved")
parser.add_argument("resume", help="specify whether it's a train/test/val split", optional)

args = parser.parse_args()

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

def main():    
    data_path = args.embedding_source
    data = [f"{data_path}/{x}" for x in os.listdir(data_path)]

    val_data = [f"new_test_cars_embeddings/{x}" for x in os.listdir('new_test_cars_embeddings')]


    balanced_data = pd.DataFrame({'data':data})
    val_data_frame = pd.DataFrame({"data":val_data})

    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Load your dataframe

    # Split into train and test sets
    train_df, test_df = train_test_split(balanced_data, test_size=0.2, random_state=42)

    # Split train set into train and validation sets
    # train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)


    partial_validation, test = train_test_split(train_df, test_size=0.2, random_state=42)
    partial_validation.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    print(train_df.shape)
    train_df.head()

    val_data_frame.head()

    train_loader = DataGenerator(balanced_data, batch_size=512)
    val_loader = DataGenerator(val_data_frame, batch_size=512)
    test_loader = DataGenerator(test, batch_size=512)


    from tqdm import tqdm


    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
    model.add(Conv1D(filters=256, kernel_size=5, activation='relu', padding='same', input_shape=(1536,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer='l2', bias_regularizer='l2'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',precision_m])

    model.load_weights('last.h5')



    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    mcp = tf.keras.callbacks.ModelCheckpoint(
        filepath='1-9-conv1d.h5',
        save_weights_only=True,
        monitor='val_precision_m',
        mode='max',
        save_best_only=True)
    import datetime
    log_dir = f"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    for i,l in train_loader:
        print(i.shape)
        break

    model.fit(train_loader, validation_data=val_loader, epochs = 20, callbacks=[mcp,tensorboard_callback,es])

if __name__ == "__main__":
    main()