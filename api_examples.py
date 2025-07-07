import requests
import json


def test_api():
    base_url = "http://localhost:5000"

    # Test data
    test_reviews = [
        {
            "text": "This product is amazing! I love it so much.",
            "model": "ml_model"
        },
        {
            "text": "Terrible quality, waste of money.",
            "model": "openai"
        },
        {
            "text": "It's okay, nothing special but does the job.",
            "model": "claude"
        }
    ]

    print("Testing API endpoints...")
    print("="*50)

    for i, review in enumerate(test_reviews, 1):
        print(f"\nTest {i}:")
        print(f"Review: {review['text']}")
        print(f"Model: {review['model']}")

        try:
            response = requests.post(
                f"{base_url}/api/predict",
                json=review,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Sentiment: {result['sentiment']}")
                print(f"✅ Confidence: {result['confidence']:.2%}")
                print(f"✅ Model Used: {result['model_used']}")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"❌ Message: {response.text}")

        except Exception as e:
            print(f"❌ Exception: {e}")

        print("-" * 30)


if __name__ == "__main__":
    test_api()
