import tensorflow as tf
import joblib
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
loadedModel = keras.models.load_model('model.h5')  

# Load the saved tokenizer
tokenizer = joblib.load('tokenizer.bin') 

# Preprocess user input
def preprocess_input(user_input_text):
    user_input_text = [user_input_text]  
    sequences = tokenizer.texts_to_sequences(user_input_text)
    paddedSequences = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
    return np.array(paddedSequences) 

# User Input
user_input = input("Enter a text: ")

# Run toxicity detection
user_input_array = preprocess_input(user_input)
toxicityPredictions = loadedModel.predict(user_input_array)

# Interpret the results
toxicityThreshold = 0.5  # Adjust as needed
toxic = any(toxicityPredictions[0] > toxicityThreshold)

if toxic:
    print("The input contains toxic language.")
else:
    print("The input is not toxic.")

print("Toxicity Scores:")
print("Toxic:", toxicityPredictions[0][0])
print("Severe Toxic:", toxicityPredictions[0][1])

