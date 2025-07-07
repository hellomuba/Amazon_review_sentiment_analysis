import os
import requests
import pickle


def download_models():
    """Download models from cloud storage on startup"""
    model_urls = {
        'sentiment_model.pkl': 'YOUR_CLOUD_STORAGE_URL/sentiment_model.pkl',
        'tfidf_vectorizer.pkl': 'YOUR_CLOUD_STORAGE_URL/tfidf_vectorizer.pkl',
        'label_encoder.pkl': 'YOUR_CLOUD_STORAGE_URL/label_encoder.pkl'
    }

    os.makedirs('models', exist_ok=True)

    for filename, url in model_urls.items():
        filepath = f'models/{filename}'
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            response = requests.get(url)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {filename}")


if __name__ == "__main__":
    download_models()
