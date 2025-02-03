import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
import matplotlib.pyplot as plt
from sklearn.utils.validation import joblib


len(tf.config.list_physical_devices('GPU'))

# Loading the dataset
# Dataset https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data
DATASET_DIR = 'dataset'
train_df = pd.read_csv(f'{DATASET_DIR}/train.csv')
test_df = pd.read_csv(f'{DATASET_DIR}/test.csv')

# Preprocess data
trainSentences = train_df['comment_text'].values
trainSentences = [str(sentence) for sentence in trainSentences]
trainLabels = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
testSentences = test_df['comment_text'].values

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(trainSentences)

trainSequences = tokenizer.texts_to_sequences(trainSentences)
trainPadded = pad_sequences(trainSequences, maxlen=200, padding='post', truncating='post')

testSequences = tokenizer.texts_to_sequences(testSentences)
testPadded = pad_sequences(testSequences, maxlen=200, padding='post', truncating='post')

# Building the model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dense(6, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Training the model
history = model.fit(trainPadded, trainLabels, epochs=5, batch_size=32, validation_split=0.2)

plt.plot(history.history['loss'], 'r', label='Training loss')
plt.plot(history.history['val_loss'], 'g', label='Validation loss')
plt.title('Training VS Validation loss')
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# print(history.history.keys())
plt.plot(history.history['accuracy'], 'r', label='Training accuracy')
plt.plot(history.history['val_accuracy'], 'g', label='Validation accuracy')
# plt.plot(history.history['binary_accuracy'], 'r', label='Training accuracy')
# plt.plot(history.history['val_binary_accuracy'], 'g', label='Validation accuracy')
plt.title('Training Vs Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predict on test set
test_pred = model.predict(testPadded)
#
# Save the predictions to CSV
submission_df = pd.read_csv(f'{DATASET_DIR}/sample_submission.csv')
submission_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = test_pred
submission_df.to_csv('submission.csv', index=False)

joblib.dump(tokenizer, 'tokenizer.bin')

model.save('model.h5')
