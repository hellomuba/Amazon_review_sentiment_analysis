from flask import Flask, render_template, request, jsonify
import os
import re
import requests
import json
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)


def enhanced_sentiment_analysis(text):
    """Enhanced rule-based sentiment analysis"""
    positive_words = [
        'good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'wonderful',
        'fantastic', 'awesome', 'outstanding', 'brilliant', 'superb', 'satisfied',
        'happy', 'recommend', 'quality', 'fast', 'beautiful', 'works', 'pleased',
        'nice', 'best', 'impressive', 'solid', 'reliable', 'efficient', 'smooth',
        'comfortable', 'stylish', 'convenient', 'helpful', 'worth', 'value'
    ]

    negative_words = [
        'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointed',
        'waste', 'poor', 'useless', 'broken', 'defective', 'angry', 'frustrated',
        'regret', 'slow', 'expensive', 'cheap', 'fake', 'damaged', 'annoying',
        'difficult', 'confusing', 'uncomfortable', 'unreliable', 'flimsy', 'overpriced'
    ]

    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)

    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)

    total_words = len(words)
    if total_words == 0:
        return 'neutral', 0.5

    if positive_count > negative_count:
        confidence = min(0.65 + (positive_count / total_words) * 2, 0.95)
        return 'positive', confidence
    elif negative_count > positive_count:
        confidence = min(0.65 + (negative_count / total_words) * 2, 0.95)
        return 'negative', confidence
    else:
        return 'neutral', 0.5


def predict_with_openai(text):
    """Use OpenAI API for sentiment analysis"""
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        return "OpenAI not configured", 0.0

    try:
        headers = {
            'Authorization': f'Bearer {openai_key}',
            'Content-Type': 'application/json'
        }

        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "Analyze the sentiment of the following text and respond with only one word: positive, negative, or neutral"},
                {"role": "user", "content": f"Text: {text}"}
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }

        response = requests.post('https://api.openai.com/v1/chat/completions',
                                 headers=headers, json=data, timeout=30)

        if response.status_code == 200:
            result = response.json()[
                'choices'][0]['message']['content'].strip().lower()

            if 'positive' in result:
                return 'positive', 0.87
            elif 'negative' in result:
                return 'negative', 0.87
            else:
                return 'neutral', 0.82
        else:
            return f"OpenAI Error: {response.status_code}", 0.0

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"OpenAI Error: {str(e)}", 0.0


def predict_with_claude(text):
    """Use Anthropic Claude for sentiment analysis"""
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_key:
        return "Claude not configured", 0.0

    try:
        headers = {
            'x-api-key': anthropic_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }

        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": f"Analyze the sentiment of this text and respond with only one word: positive, negative, or neutral. Text: {text}"}
            ]
        }

        response = requests.post('https://api.anthropic.com/v1/messages',
                                 headers=headers, json=data, timeout=30)

        if response.status_code == 200:
            result = response.json()['content'][0]['text'].strip().lower()

            if 'positive' in result:
                return 'positive', 0.89
            elif 'negative' in result:
                return 'negative', 0.89
            else:
                return 'neutral', 0.84
        else:
            return f"Claude Error: {response.status_code}", 0.0

    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return f"Claude Error: {str(e)}", 0.0


def predict_with_gemini(text):
    """Use Google Gemini for sentiment analysis"""
    google_key = os.getenv('GOOGLE_API_KEY')
    if not google_key:
        return "Gemini not configured", 0.0

    # Use the available models from your API key
    models_to_try = [
        "models/gemini-1.5-flash",      # Fast and efficient
        "models/gemini-1.5-pro",        # Higher quality
        "models/gemini-2.0-flash",      # Newest model
        "models/gemini-1.5-flash-002",  # Alternative version
        "models/gemini-2.5-flash"       # Another option
    ]

    for model_name in models_to_try:
        try:
            url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={google_key}"

            headers = {
                'Content-Type': 'application/json'
            }

            data = {
                "contents": [{
                    "parts": [{
                        "text": f"Analyze the sentiment of this text and respond with only one word: positive, negative, or neutral. Text: {text}"
                    }]
                }],
                "generationConfig": {
                    "maxOutputTokens": 10,
                    "temperature": 0.1
                }
            }

            response = requests.post(
                url, headers=headers, json=data, timeout=30)

            if response.status_code == 200:
                result_data = response.json()
                if 'candidates' in result_data and len(result_data['candidates']) > 0:
                    candidate = result_data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        result = candidate['content']['parts'][0]['text'].strip(
                        ).lower()

                        logger.info(
                            f"Gemini ({model_name}) response: {result}")

                        if 'positive' in result:
                            return 'positive', 0.85
                        elif 'negative' in result:
                            return 'negative', 0.85
                        else:
                            return 'neutral', 0.80
                    else:
                        logger.warning(
                            f"Gemini model {model_name}: No content in response")
                        continue
                else:
                    logger.warning(
                        f"Gemini model {model_name}: No candidates in response")
                    continue
            else:
                logger.warning(
                    f"Gemini model {model_name} failed with status {response.status_code}: {response.text}")
                continue

        except Exception as e:
            logger.error(f"Gemini API error with model {model_name}: {e}")
            continue

    return "Gemini Error: All models failed", 0.0

# ... (rest of your app routes remain the same)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        review_text = request.form.get('review_text', '').strip()
        ai_model = request.form.get('ai_model', 'enhanced')

        if not review_text:
            return render_template('result.html', error="Please enter a review text.")

        if len(review_text) < 3:
            return render_template('result.html', error="Please enter a more meaningful review (at least 3 characters).")

        # Choose prediction method
        if ai_model == 'openai':
            sentiment, confidence = predict_with_openai(review_text)
            model_used = "OpenAI GPT-3.5"
        elif ai_model == 'claude':
            sentiment, confidence = predict_with_claude(review_text)
            model_used = "Anthropic Claude"
        elif ai_model == 'gemini':
            sentiment, confidence = predict_with_gemini(review_text)
            model_used = "Google Gemini"
        else:
            sentiment, confidence = enhanced_sentiment_analysis(review_text)
            model_used = "Enhanced Rule-Based Analysis"

        # Handle errors
        if isinstance(sentiment, str) and ("Error" in sentiment or "not configured" in sentiment):
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

        if len(review_text) < 3:
            return render_template('compare.html', error="Please enter a more meaningful review (at least 3 characters).")

        # Get predictions from all available models
        results = {}

        # Enhanced Rule-Based Model (always available)
        enhanced_sentiment, enhanced_confidence = enhanced_sentiment_analysis(
            review_text)
        results['Enhanced Analysis'] = {
            'sentiment': enhanced_sentiment,
            'confidence': enhanced_confidence,
            'model_name': "Enhanced Rule-Based Model"
        }

        # OpenAI
        openai_sentiment, openai_confidence = predict_with_openai(review_text)
        results['OpenAI'] = {
            'sentiment': openai_sentiment,
            'confidence': openai_confidence,
            'model_name': "GPT-3.5 Turbo"
        }

        # Claude
        claude_sentiment, claude_confidence = predict_with_claude(review_text)
        results['Claude'] = {
            'sentiment': claude_sentiment,
            'confidence': claude_confidence,
            'model_name': "Claude 3 Haiku"
        }

        # Gemini
        gemini_sentiment, gemini_confidence = predict_with_gemini(review_text)
        results['Gemini'] = {
            'sentiment': gemini_sentiment,
            'confidence': gemini_confidence,
            'model_name': "Gemini 1.5 Flash"
        }

        return render_template('compare.html',
                               review_text=review_text,
                               results=results)

    except Exception as e:
        logger.error(f"Comparison error: {e}")
        return render_template('compare.html',
                               error=f"An error occurred: {str(e)}")


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/debug-gemini')
def debug_gemini():
    """Debug Gemini API issues"""
    google_key = os.getenv('GOOGLE_API_KEY')
    if not google_key:
        return jsonify({'error': 'No Google API key found'})

    try:
        # List available models
        url = f"https://generativelanguage.googleapis.com/v1/models?key={google_key}"
        response = requests.get(url, timeout=30)

        test_results = {
            'api_key_length': len(google_key),
            'api_key_starts_with': google_key[:10] + '...' if len(google_key) > 10 else google_key,
            'list_models_status': response.status_code,
        }

        if response.status_code == 200:
            models = response.json()
            all_models = []
            generateContent_models = []

            for model in models.get('models', []):
                all_models.append(model['name'])
                if 'generateContent' in model.get('supportedGenerationMethods', []):
                    generateContent_models.append(model['name'])

            test_results.update({
                'all_models': all_models,
                'generateContent_models': generateContent_models,
                'total_models': len(all_models),
                'supported_models': len(generateContent_models)
            })
        else:
            test_results['error'] = f"Failed to list models: {response.status_code} - {response.text}"

        return jsonify(test_results)

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Application is running',
        'api_keys_found': {
            'openai': bool(os.getenv('OPENAI_API_KEY')),
            'claude': bool(os.getenv('ANTHROPIC_API_KEY')),
            'gemini': bool(os.getenv('GOOGLE_API_KEY'))
        }
    }), 200


if __name__ == '__main__':
    logger.info("🚀 Starting Amazon Review Sentiment Analysis")
    logger.info("="*50)
    logger.info(
        f"OpenAI API Key: {'✅ Found' if os.getenv('OPENAI_API_KEY') else '❌ Not found'}")
    logger.info(
        f"Anthropic API Key: {'✅ Found' if os.getenv('ANTHROPIC_API_KEY') else '❌ Not found'}")
    logger.info(
        f"Google API Key: {'✅ Found' if os.getenv('GOOGLE_API_KEY') else '❌ Not found'}")
    logger.info("="*50)
    logger.info("🌐 Flask application starting...")
    logger.info("📱 Open: http://localhost:5000")
    logger.info("="*50)

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
