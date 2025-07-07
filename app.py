from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from dotenv import load_dotenv
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

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
                logger.info(f"Downloading {item}...")
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

# AI API clients
openai_client = None
anthropic_client = None
gemini_model = None


def initialize_ai_clients():
    """Initialize AI clients with proper error handling"""
    global openai_client, anthropic_client, gemini_model

    # Initialize OpenAI
    try:
        if os.getenv('OPENAI_API_KEY'):
            import openai
            openai.api_key = os.getenv('OPENAI_API_KEY')
            openai_client = openai
            logger.info("✅ OpenAI client initialized")
        else:
            logger.info("⚠️ OpenAI API key not found")
    except Exception as e:
        logger.error(f"❌ OpenAI initialization failed: {e}")

    # Initialize Anthropic
    try:
        if os.getenv('ANTHROPIC_API_KEY'):
            import anthropic
            anthropic_client = anthropic.Anthropic(
                api_key=os.getenv('ANTHROPIC_API_KEY'))
            logger.info("✅ Anthropic client initialized")
        else:
            logger.info("⚠️ Anthropic API key not found")
    except Exception as e:
        logger.error(f"❌ Anthropic initialization failed: {e}")

    # Initialize Google Gemini
    try:
        if os.getenv('GOOGLE_API_KEY'):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            gemini_model = genai.GenerativeModel('gemini-pro')
            logger.info("✅ Google Gemini client initialized")
        else:
            logger.info("⚠️ Google API key not found")
    except Exception as e:
        logger.error(f"❌ Gemini initialization failed: {e}")


class AIPredictor:
    def predict_with_openai(self, text):
        if not openai_client:
            return "OpenAI not configured", 0.0

        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Analyze sentiment and respond with only: positive, negative, or neutral"},
                    {"role": "user", "content": f"Review: {text}"}
                ],
                max_tokens=10,
                temperature=0.3
            )
            result = response.choices[0].message.content.strip().lower()

            if 'positive' in result:
                return 'positive', 0.85
            elif 'negative' in result:
                return 'negative', 0.85
            else:
                return 'neutral', 0.80
        except Exception as e:
            return f"OpenAI Error: {str(e)}", 0.0

    def predict_with_claude(self, text):
        if not anthropic_client:
            return "Claude not configured", 0.0

        try:
            response = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=10,
                messages=[
                    {"role": "user",
                        "content": f"Sentiment of this review (positive/negative/neutral): {text}"}
                ]
            )
            result = response.content[0].text.strip().lower()

            if 'positive' in result:
                return 'positive', 0.87
            elif 'negative' in result:
                return 'negative', 0.87
            else:
                return 'neutral', 0.82
        except Exception as e:
            return f"Claude Error: {str(e)}", 0.0

    def predict_with_gemini(self, text):
        if not gemini_model:
            return "Gemini not configured", 0.0

        try:
            response = gemini_model.generate_content(
                f"Sentiment (positive/negative/neutral): {text}")
            result = response.text.strip().lower()

            if 'positive' in result:
                return 'positive', 0.83
            elif 'negative' in result:
                return 'negative', 0.83
            else:
                return 'neutral', 0.78
        except Exception as e:
            return f"Gemini Error: {str(e)}", 0.0


ai_predictor = AIPredictor()


def create_sample_data():
    """Create sample data if models not available"""
    return {
        'positive': ['This product is amazing! Great quality and fast shipping.',
                     'Excellent service and fantastic product quality!'],
        'negative': ['Terrible quality, waste of money. Very disappointed.',
                     'Poor customer service and defective product.'],
        'neutral': ['The product is okay. Average quality for the price.',
                    'It works as expected but nothing special.']
    }


def load_models():
    """Load trained models with fallback"""
    global model, tfidf_vectorizer, label_encoder, model_metadata

    try:
        if not os.path.exists('models'):
            logger.warning(
                "⚠️ Models directory not found - using AI APIs only")
            return False

        with open('models/sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        if os.path.exists('models/label_encoder.pkl'):
            with open('models/label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)

        if os.path.exists('models/model_metadata.pkl'):
            with open('models/model_metadata.pkl', 'rb') as f:
                model_metadata = pickle.load(f)

        logger.info("✅ ML models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return False


def preprocess_text(text):
    """Preprocess text for ML model"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(
        word) for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)


def predict_sentiment_ml(text):
    """Predict sentiment using trained ML model"""
    if not model or not tfidf_vectorizer:
        return "ML models not available", 0.0

    try:
        processed_text = preprocess_text(text)
        text_tfidf = tfidf_vectorizer.transform([processed_text])
        prediction = model.predict(text_tfidf)[0]
        probability = model.predict_proba(text_tfidf)[0]

        if label_encoder:
            predicted_label = label_encoder.inverse_transform([prediction])[0]
        else:
            predicted_label = prediction

        confidence = max(probability)
        return predicted_label, confidence
    except Exception as e:
        return f"ML Error: {e}", 0.0


def predict_sentiment_fallback(text):
    """Simple rule-based fallback prediction"""
    positive_words = ['good', 'great', 'excellent',
                      'amazing', 'love', 'perfect', 'wonderful', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'hate',
                      'worst', 'horrible', 'disappointed', 'waste']

    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if positive_count > negative_count:
        return 'positive', 0.65
    elif negative_count > positive_count:
        return 'negative', 0.65
    else:
        return 'neutral', 0.60


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        review_text = request.form.get('review_text', '').strip()
        ai_model = request.form.get('ai_model', 'ml_model')

        if not review_text:
            return render_template('result.html', error="Please enter a review text.")

        # Choose prediction method
        if ai_model == 'openai':
            sentiment, confidence = ai_predictor.predict_with_openai(
                review_text)
            model_used = "OpenAI GPT-3.5"
        elif ai_model == 'claude':
            sentiment, confidence = ai_predictor.predict_with_claude(
                review_text)
            model_used = "Anthropic Claude"
        elif ai_model == 'gemini':
            sentiment, confidence = ai_predictor.predict_with_gemini(
                review_text)
            model_used = "Google Gemini"
        else:
            sentiment, confidence = predict_sentiment_ml(review_text)
            if "not available" in sentiment or "Error" in sentiment:
                sentiment, confidence = predict_sentiment_fallback(review_text)
                model_used = "Rule-based Fallback"
            else:
                model_used = model_metadata['model_name'] if model_metadata else "ML Model"

        if isinstance(sentiment, str) and ("Error" in sentiment and "not configured" not in sentiment):
            return render_template('result.html', error=sentiment)

        result = {
            'review_text': review_text,
            'sentiment': sentiment,
            'confidence': f"{confidence:.2%}",
            'confidence_score': confidence,
            'model_used': model_used
        }

        return render_template('result.html', result=result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template('result.html', error=f"An error occurred: {str(e)}")


@app.route('/compare', methods=['POST'])
def compare_models():
    """Compare predictions from different AI models"""
    try:
        review_text = request.form.get('review_text', '').strip()

        if not review_text:
            return render_template('compare.html', error="Please enter a review text.")

        results = {}

        # ML Model
        ml_sentiment, ml_confidence = predict_sentiment_ml(review_text)
        if "not available" in ml_sentiment or "Error" in ml_sentiment:
            ml_sentiment, ml_confidence = predict_sentiment_fallback(
                review_text)
            model_name = "Rule-based Fallback"
        else:
            model_name = model_metadata['model_name'] if model_metadata else "ML Model"

        results['ML Model'] = {
            'sentiment': ml_sentiment,
            'confidence': ml_confidence,
            'model_name': model_name
        }

        # AI Models (only if configured)
        if openai_client:
            openai_sentiment, openai_confidence = ai_predictor.predict_with_openai(
                review_text)
            results['OpenAI'] = {
                'sentiment': openai_sentiment,
                'confidence': openai_confidence,
                'model_name': "GPT-3.5 Turbo"
            }

        if anthropic_client:
            claude_sentiment, claude_confidence = ai_predictor.predict_with_claude(
                review_text)
            results['Claude'] = {
                'sentiment': claude_sentiment,
                'confidence': claude_confidence,
                'model_name': "Claude 3 Sonnet"
            }

        if gemini_model:
            gemini_sentiment, gemini_confidence = ai_predictor.predict_with_gemini(
                review_text)
            results['Gemini'] = {
                'sentiment': gemini_sentiment,
                'confidence': gemini_confidence,
                'model_name': "Gemini Pro"
            }

        return render_template('compare.html', review_text=review_text, results=results)

    except Exception as e:
        logger.error(f"Comparison error: {e}")
        return render_template('compare.html', error=f"An error occurred: {str(e)}")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Please provide text in JSON format'}), 400

        review_text = data['text'].strip()
        ai_model = data.get('model', 'ml_model')

        if not review_text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        # Choose prediction method
        if ai_model == 'openai':
            sentiment, confidence = ai_predictor.predict_with_openai(
                review_text)
            model_used = "OpenAI GPT-3.5"
        elif ai_model == 'claude':
            sentiment, confidence = ai_predictor.predict_with_claude(
                review_text)
            model_used = "Anthropic Claude"
        elif ai_model == 'gemini':
            sentiment, confidence = ai_predictor.predict_with_gemini(
                review_text)
            model_used = "Google Gemini"
        else:
            sentiment, confidence = predict_sentiment_ml(review_text)
            if "not available" in sentiment or "Error" in sentiment:
                sentiment, confidence = predict_sentiment_fallback(review_text)
                model_used = "Rule-based Fallback"
            else:
                model_used = model_metadata['model_name'] if model_metadata else "ML Model"

        # Check if there was an error
        if isinstance(sentiment, str) and ("Error" in sentiment and "not configured" not in sentiment):
            return jsonify({'error': sentiment}), 500

        return jsonify({
            'text': review_text,
            'sentiment': sentiment,
            'confidence': confidence,
            'model_used': model_used
        })

    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/health')
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({'status': 'healthy', 'message': 'Application is running'}), 200


if __name__ == '__main__':
    logger.info("🚀 Starting Amazon Review Sentiment Analysis")
    logger.info("="*50)

    # Initialize AI clients
    initialize_ai_clients()

    # Load ML models
    load_models()

    logger.info("="*50)
    logger.info("🌐 Flask application starting...")

    # Use environment variable for port (Railway requirement)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
