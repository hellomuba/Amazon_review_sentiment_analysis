from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download required NLTK data


def download_nltk_data():
    nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for item in nltk_downloads:
        try:
            nltk.data.find(f'tokenizers/{item}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{item}')
            except LookupError:
                nltk.download(item)


download_nltk_data()

app = Flask(__name__)

# Initialize global variables
model = None
tfidf_vectorizer = None
label_encoder = None
model_metadata = None
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def load_models():
    """Load trained models and preprocessors"""
    global model, tfidf_vectorizer, label_encoder, model_metadata

    try:
        # Load the trained model
        with open('models/sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Load the TF-IDF vectorizer
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        # Load the label encoder if it exists
        if os.path.exists('models/label_encoder.pkl'):
            with open('models/label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)

        # Load model metadata
        if os.path.exists('models/model_metadata.pkl'):
            with open('models/model_metadata.pkl', 'rb') as f:
                model_metadata = pickle.load(f)

        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


def preprocess_text(text):
    """Preprocess text for ML model"""
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and remove stopwords
    words = text.split()
    words = [lemmatizer.lemmatize(
        word) for word in words if word not in stop_words and len(word) > 2]

    return ' '.join(words)


def predict_sentiment_ml(text):
    """Predict sentiment using trained ML model"""
    if not model or not tfidf_vectorizer:
        return "Error: Models not loaded", 0.0

    try:
        # Preprocess the text
        processed_text = preprocess_text(text)

        # Vectorize
        text_tfidf = tfidf_vectorizer.transform([processed_text])

        # Predict
        prediction = model.predict(text_tfidf)[0]
        probability = model.predict_proba(text_tfidf)[0]

        # Decode label if necessary
        if label_encoder:
            predicted_label = label_encoder.inverse_transform([prediction])[0]
        else:
            predicted_label = prediction

        confidence = max(probability)

        return predicted_label, confidence

    except Exception as e:
        return f"Error in ML prediction: {e}", 0.0


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get the review text from the form
        review_text = request.form.get('review_text', '').strip()
        ai_model = request.form.get('ai_model', 'ml_model')

        if not review_text:
            return render_template('result.html',
                                   error="Please enter a review text.")

        # For now, only use ML model (you can add AI APIs later)
        sentiment, confidence = predict_sentiment_ml(review_text)
        model_used = model_metadata['model_name'] if model_metadata else "ML Model"

        # Check if there was an error
        if isinstance(sentiment, str) and ("Error" in sentiment):
            return render_template('result.html',
                                   error=sentiment)

        # Prepare result
        result = {
            'review_text': review_text,
            'sentiment': sentiment,
            'confidence': f"{confidence:.2%}",
            'confidence_score': confidence,
            'model_used': model_used
        }

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('result.html',
                               error=f"An error occurred: {str(e)}")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Please provide text in JSON format'}), 400

        review_text = data['text'].strip()

        if not review_text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        # Make prediction
        sentiment, confidence = predict_sentiment_ml(review_text)
        model_used = model_metadata['model_name'] if model_metadata else "ML Model"

        # Check if there was an error
        if isinstance(sentiment, str) and ("Error" in sentiment):
            return jsonify({'error': sentiment}), 500

        return jsonify({
            'text': review_text,
            'sentiment': sentiment,
            'confidence': confidence,
            'model_used': model_used
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


if __name__ == '__main__':
    # Load models on startup
    if load_models():
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Please run sentiment_analysis.py first to train the models.")
