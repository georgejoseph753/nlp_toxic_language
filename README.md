NLP Toxic Language Detection

This project is a Toxic Language Detection Model designed to help social media platforms create a safer and more refined environment by identifying and analyzing toxic comments.
Project Overview

The model is trained using Natural Language Processing (NLP) techniques to detect toxicity in user comments.
The training script nlp.py is used to build and train the model.
The trained model is saved as model.h5 and later loaded in model_load_nlp.py for testing and evaluation.
The model can analyze user input text and classify its toxicity level and severity.

Installation & Setup

Clone the repository:

git clone https://github.com/georgejoseph753/nlp_toxic_language.git
cd nlp_toxic_language

Install dependencies:

pip install -r requirements.txt

Run the model for testing:

python model_load_nlp.py

Usage

The user provides a text input through model_load_nlp.py, and the model analyzes the toxicity level of the text.
The output will indicate if the comment is toxic, severe toxic, obscene, threatening, insulting, or identity hate.

Files in this Repository

nlp.py → Trains the NLP model and saves it as model.h5.
model.h5 → pre-trained model used for toxicity detection.
model_load_nlp.py → Loads the model and performs toxicity analysis on user input.

Future Improvements

Enhance accuracy by training with larger datasets.
Implement an API for real-time toxicity detection.
Improve classification by adding context-based sentiment analysis.

License

This project is open-source and available under the MIT License.
