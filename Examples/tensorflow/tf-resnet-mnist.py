import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Dropout,Flatten,Add
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os

#Settings
batch_size = 128
epochs = 50
patience=3            # Stop training if loss doesn't improve for this many epochs
log_dir="./log"

# Create log dir if necessary
if not os.path.exists(log_dir):
   os.makedirs(log_dir)

# input image dimensions
img_rows, img_cols = 28, 28

# The data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Features has to be in the following shape: (obs, rows, cols, color channels)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# Normalize features
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Model
inputs = Input(shape=input_shape)
x = Conv2D(6, kernel_size=(5, 5),
              activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)

# A residual block
x1 = Conv2D(6, kernel_size=(5, 5),
               activation='relu',
               padding='same')(x)
x2 = Conv2D(6, kernel_size=(5, 5),
               activation='relu',
               padding='same')(x1)
x = Add()([x,x2])

# The rest is the same as before
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(120, activation='relu')(x)
x = Dense(100, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for early stopping and tensorboard
callbacks = [EarlyStopping(monitor='loss', patience=patience),
            TensorBoard(log_dir=log_dir)]

# Training
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])