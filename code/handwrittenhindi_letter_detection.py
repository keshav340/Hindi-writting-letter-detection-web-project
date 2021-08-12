

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

df = pd.read_csv('/content/drive/MyDrive/data.csv')

# From First Col to (last-1) col with all rows
X = df.iloc[:,:-1]

# last Col with all rows
y = df.iloc[:,-1]

X_images = X.values.reshape(92000,32,32)
import matplotlib.pyplot as plt
plt.imshow(X_images[0])
plt.show()

# output in binary format for NN
from sklearn.preprocessing import LabelBinarizer
binencoder = LabelBinarizer()
y = binencoder.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size=0.20, random_state=92)
X_train = X_train/255
X_test = X_test/255

# changing from 3 to 4 dimensions of inputss
X_train = X_train.reshape(X_train.shape[0], 32, 32, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 32, 32, 1).astype('float32')

# Building the Convolutional Model
conv_model = Sequential()
conv_model.add(
    Conv2D(32, (4, 4), 
           input_shape=(32, 32,1),
           activation='relu', 
           name="firstConv"
    )
)

conv_model.add(
    MaxPooling2D(pool_size=(2, 2), 
                 name="FirstPool"
                )
)

conv_model.add(
    Conv2D(64, (3, 3), 
           activation='relu', 
           name="SecondConv"
          )
)

conv_model.add(
    MaxPooling2D(
        pool_size=(2, 2),
        name="SecondPool")
)

conv_model.add(Dropout(0.2)) # Prevents Overfitting in Conv Nete
conv_model.add(Flatten())

conv_model.add(
    Dense(128,
          activation='relu',
          name="dense_1"
         )
)
conv_model.add(
    Dense(50, 
          activation='relu', 
          name="dense_2"
         )
)

# Output Layer with 46 Unique Outputs
conv_model.add(
    Dense(46, 
          activation='softmax', 
          name="modeloutput"
         )
)
conv_model.summary()

conv_model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=['accuracy']
)

from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

                     
checkpoint = ModelCheckpoint("hindi_handwritting_detection.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)
callbacks = [earlystop, checkpoint]

result = conv_model.fit(X_train, y_train, validation_split=0.20, epochs=10, batch_size=92,verbose=2,callbacks = callbacks)

scores = conv_model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))

num = 15000
plt.imshow(X_images[num])
plt.show()

loss_train = result.history['loss']
loss_valid = result.history['val_loss']
acc_train = result.history['accuracy']
acc_valid = result.history['val_accuracy']

epochs = 10

#how to predict
imgTrans = X_images[num].reshape(1,32,32,1)
imgTrans.shape

predictions = conv_model.predict(imgTrans)
binencoder.classes_[np.argmax(predictions)]
