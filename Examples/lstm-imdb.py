from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from sklearn.utils import resample
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Embedding
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# Settings
max_features = 20000  # How many words to keep?
maxlen = 80           # cut texts after this number of words
n_samples = None      # None means full sample
batch_size = 128
epochs = 30
patience=3            # Stop training if loss doesn't improve for this many epochs

# Load data
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
if (n_samples != None):
    x_train,y_train,x_test,y_test = resample(x_train,y_train,
                                             x_test,y_test,
                                             n_samples=n_samples)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# Data processing
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Model
print('Build model...')
inputs = Input(shape=(maxlen,))
x = Embedding(max_features, 128)(inputs)
x = Bidirectional(LSTM(128, dropout=0.2))(x)
x = Dense(128)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=output)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

# Callback for early stopping
callback = EarlyStopping(monitor='loss', patience=patience)

# Training
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[callback],
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)