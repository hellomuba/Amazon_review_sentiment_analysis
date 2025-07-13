import os
import sys
from sentiment_analysis import AmazonSentimentAnalyzer


def main():
    print("="*60)
    print("Amazon Review Sentiment Analysis - Setup")
    print("="*60)

    # Check if models exist
    if not os.path.exists('models/sentiment_model.pkl'):
        print("Models not found. Training models first...")
        analyzer = AmazonSentimentAnalyzer()
        analyzer.run_complete_pipeline()
    else:
        print("Models found. Skipping training...")

    # Start Flask app
    print("\nStarting Flask application...")
    from app import app
    app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == "__main__":
    main()
