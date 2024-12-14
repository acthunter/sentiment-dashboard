from flask import Flask, render_template, request
import pandas as pd
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Setup Flask
app = Flask(__name__)

# Load lexicons
positive_words = set(pd.read_csv(os.path.join(os.getcwd(), 'content', 'positive.csv'))['word'].tolist())
negative_words = set(pd.read_csv(os.path.join(os.getcwd(), 'content', 'negative.csv'))['word'].tolist())

# Load slang normalization dictionary
slang_file_path = os.path.join(os.getcwd(), 'content', 'Slangword-indonesian.csv')
slang_dict = pd.read_csv(slang_file_path)

# Periksa kolom yang tersedia dalam file CSV
print("Kolom dalam file Slangword-indonesian.csv:", slang_dict.columns)

# Pastikan kolom yang digunakan sesuai
if 'slang' in slang_dict.columns and 'formal' in slang_dict.columns:
    slang_dict = slang_dict.set_index('slang')['formal'].to_dict()
else:
    raise ValueError("File Slangword-indonesian.csv tidak memiliki kolom 'slang' dan 'formal'.")

# Setup NLTK and Sastrawi
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Function to normalize slang words
def normalize_slang(text):
    words = text.split()
    normalized_text = ' '.join([slang_dict.get(word, word) for word in words])
    return normalized_text

# Function to process text
def process_text(text):
    # Case Folding
    text = text.lower()

    # Cleansing (remove non-alphabetical characters)
    text = re.sub(r'[^a-z\s]', '', text)

    # Normalization (slang to formal)
    text = normalize_slang(text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Sentiment analysis using lexicon
    positive_score = sum(1 for word in tokens if word in positive_words)
    negative_score = sum(1 for word in tokens if word in negative_words)
    
    if positive_score > negative_score:
        return 'positif'
    elif positive_score < negative_score:
        return 'negatif'
    else:
        return 'netral'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentiment', methods=['POST'])
def sentiment():
    # Get the input text from the form
    text_input = request.form['text_input']

    # Process text to get sentiment
    sentiment_label = process_text(text_input)

    # Render the result template with the sentiment
    return render_template('result.html', sentiment=sentiment_label, text_input=text_input)

if __name__ == '__main__':
    app.run(debug=True)
