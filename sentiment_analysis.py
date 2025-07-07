import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import warnings
import os
from datasets import load_dataset
warnings.filterwarnings('ignore')

# Download required NLTK data


def download_nltk_data():
    """Download required NLTK data"""
    nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for item in nltk_downloads:
        try:
            nltk.data.find(f'tokenizers/{item}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{item}')
            except LookupError:
                print(f"Downloading {item}...")
                nltk.download(item)


download_nltk_data()


class AmazonSentimentAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None
        self.best_model = None
        self.label_encoder = None
        self.model_name = None

    def load_data(self):
        """Load the Amazon reviews dataset"""
        print("Loading dataset...")
        try:
            # Try loading from Hugging Face datasets
            dataset = load_dataset(
                "hugginglearners/amazon-reviews-sentiment-analysis")
            df = pd.DataFrame(dataset['train'])
            print(f"Dataset loaded successfully from Hugging Face!")
        except Exception as e:
            print(f"Error loading from Hugging Face: {e}")
            try:
                # Fallback to direct CSV loading
                df = pd.read_csv(
                    "hf://datasets/hugginglearners/amazon-reviews-sentiment-analysis/amazon_reviews.csv")
                print(f"Dataset loaded successfully from CSV!")
            except Exception as e2:
                print(f"Error loading CSV: {e2}")
                # Create sample data for demonstration
                df = self.create_sample_data()
                print("Using sample data for demonstration...")

        return df

    def create_sample_data(self):
        """Create sample data if dataset loading fails"""
        sample_data = {
            'review_text': [
                "This product is amazing! I love it so much. Great quality and fast shipping.",
                "Terrible quality, waste of money. Very disappointed with this purchase.",
                "It's okay, nothing special but does the job. Average quality for the price.",
                "Excellent product! Highly recommend. Worth every penny.",
                "Poor customer service and defective product. Would not buy again.",
                "Good value for money. Works as expected. Satisfied with purchase.",
                "Outstanding quality! Exceeded my expectations. Five stars!",
                "Cheap material, broke after one day. Complete waste of money.",
                "Decent product. Not the best but not the worst either.",
                "Fantastic! Best purchase I've made. Absolutely love it!"
            ] * 100,  # Repeat to create more samples
            'sentiment': [
                'positive', 'negative', 'neutral', 'positive', 'negative',
                'positive', 'positive', 'negative', 'neutral', 'positive'
            ] * 100
        }
        return pd.DataFrame(sample_data)

    def explore_data(self, df):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)

        print(f"Dataset shape: {df.shape}")
        print(f"Dataset columns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())

        print("\nDataset Info:")
        print(df.info())

        print("\nMissing values:")
        print(df.isnull().sum())

        # Check if we need to map ratings to sentiment labels
        if 'sentiment' not in df.columns and 'overall' in df.columns:
            # Map numerical ratings to sentiment labels
            def map_sentiment(rating):
                if rating >= 4:
                    return 'positive'
                elif rating == 3:
                    return 'neutral'
                else:
                    return 'negative'
            df['sentiment'] = df['overall'].apply(map_sentiment)
            print("Mapped 'overall' ratings to 'sentiment' labels.")

        # Ensure we have the required columns
        if 'sentiment' not in df.columns:
            print(
                "Warning: 'sentiment' column not found. Creating neutral sentiment for all reviews.")
            df['sentiment'] = 'neutral'

        if 'review_text' not in df.columns:
            # Try different possible column names
            text_columns = ['reviewText', 'text', 'review', 'comment']
            for col in text_columns:
                if col in df.columns:
                    df['review_text'] = df[col]
                    print(f"Using '{col}' column as review text.")
                    break
            else:
                print("Error: No text column found in the dataset.")
                return df

        print(f"\nUnique sentiment values: {df['sentiment'].unique()}")
        print(f"Sentiment distribution:")
        print(df['sentiment'].value_counts())

        # Visualizations
        plt.figure(figsize=(15, 10))

        # Sentiment distribution
        plt.subplot(2, 3, 1)
        sentiment_counts = df['sentiment'].value_counts()
        colors = ['green' if x == 'positive' else 'red' if x ==
                  'negative' else 'orange' for x in sentiment_counts.index]
        sentiment_counts.plot(kind='bar', color=colors)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=45)

        plt.subplot(2, 3, 2)
        sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors)
        plt.title('Sentiment Distribution (Pie Chart)')
        plt.ylabel('')

        # Text length analysis
        df['text_length'] = df['review_text'].str.len()
        plt.subplot(2, 3, 3)
        plt.hist(df['text_length'], bins=30, alpha=0.7, color='skyblue')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Review Text Length')

        # Box plot for text length by sentiment
        plt.subplot(2, 3, 4)
        sns.boxplot(data=df, x='sentiment', y='text_length')
        plt.title('Text Length by Sentiment')
        plt.xticks(rotation=45)

        # Word count analysis
        df['word_count'] = df['review_text'].str.split().str.len()
        plt.subplot(2, 3, 5)
        sns.boxplot(data=df, x='sentiment', y='word_count')
        plt.title('Word Count by Sentiment')
        plt.xticks(rotation=45)

        # Average text length by sentiment
        plt.subplot(2, 3, 6)
        avg_length = df.groupby('sentiment')['text_length'].mean()
        colors_avg = ['green' if x == 'positive' else 'red' if x ==
                      'negative' else 'orange' for x in avg_length.index]
        avg_length.plot(kind='bar', color=colors_avg)
        plt.title('Average Text Length by Sentiment')
        plt.xlabel('Sentiment')
        plt.ylabel('Average Length')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        return df

    def preprocess_data(self, df):
        """Clean and preprocess the data"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)

        # Remove duplicates
        initial_shape = df.shape
        df = df.drop_duplicates()
        print(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")

        # Handle missing values
        df = df.dropna(subset=['review_text', 'sentiment'])
        print(f"Final dataset shape: {df.shape}")

        # Ensure text columns are strings
        df['review_text'] = df['review_text'].astype(str)

        return df

    def preprocess_text(self, text):
        """Preprocess individual text"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize and remove stopwords
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words
                 if word not in self.stop_words and len(word) > 2]

        return ' '.join(words)

    def prepare_features(self, df):
        """Prepare features for modeling"""
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)

        # Apply text preprocessing
        print("Preprocessing text data...")
        df['processed_text'] = df['review_text'].apply(self.preprocess_text)

        # Show preprocessing examples
        print("\nPreprocessing examples:")
        for i in range(min(3, len(df))):
            print(f"\nOriginal: {df['review_text'].iloc[i][:100]}...")
            print(f"Processed: {df['processed_text'].iloc[i][:100]}...")

        # Prepare features and target
        X = df['processed_text']
        y = df['sentiment']

        # Encode labels if necessary
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            label_mapping = dict(zip(self.label_encoder.classes_,
                                     self.label_encoder.transform(self.label_encoder.classes_)))
            print(f"\nLabel mapping: {label_mapping}")
        else:
            y_encoded = y
            self.label_encoder = None

        return X, y_encoded

    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        print("\n" + "="*50)
        print("MODEL TRAINING & EVALUATION")
        print("="*50)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")

        # TF-IDF Vectorization
        print("\nApplying TF-IDF Vectorization...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)

        print(f"TF-IDF feature shape: {X_train_tfidf.shape}")

        # Define models to train
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(random_state=42, probability=True)
        }

        results = {}

        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_tfidf, y_train)

            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }

            print(f"{name} Accuracy: {accuracy:.4f}")

        # Select the best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_model_name]['model']
        self.model_name = best_model_name
        best_accuracy = results[best_model_name]['accuracy']
        best_predictions = results[best_model_name]['predictions']

        print(
            f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")

        # Detailed evaluation of the best model
        print(f"\n--- {best_model_name} Detailed Results ---")
        target_names = self.label_encoder.classes_ if self.label_encoder else None
        print(classification_report(
            y_test, best_predictions, target_names=target_names))

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

        return X_test, y_test, best_predictions

    def save_model(self):
        """Save the trained model and components"""
        print("\n" + "="*50)
        print("SAVING MODEL AND COMPONENTS")
        print("="*50)

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Save the best model
        with open('models/sentiment_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)

        # Save the TF-IDF vectorizer
        with open('models/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)

        # Save the label encoder if used
        if self.label_encoder:
            with open('models/label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)

        # Save model metadata
        metadata = {
            'model_name': self.model_name,
            'features': 5000,
            'ngram_range': (1, 2)
        }

        with open('models/model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        print("Model saved successfully!")
        print("Files created:")
        print("- models/sentiment_model.pkl")
        print("- models/tfidf_vectorizer.pkl")
        if self.label_encoder:
            print("- models/label_encoder.pkl")
        print("- models/model_metadata.pkl")

    def test_model(self):
        """Test the saved model with sample predictions"""
        print("\n" + "="*50)
        print("TESTING MODEL WITH SAMPLE PREDICTIONS")
        print("="*50)

        test_reviews = [
            "This product is amazing! I love it so much. Great quality and fast shipping.",
            "Terrible quality, waste of money. Very disappointed with this purchase.",
            "It's okay, nothing special but does the job. Average quality for the price.",
            "Excellent customer service and outstanding product quality!",
            "Poor design and materials. Broke after one week of use."
        ]

        for review in test_reviews:
            sentiment, confidence = self.predict_sentiment(review)
            print(f"Review: {review}")
            print(f"Predicted Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.2%}")
            print("-" * 80)

    def predict_sentiment(self, text):
        """Predict sentiment for a given text"""
        # Preprocess the text
        processed_text = self.preprocess_text(text)

        # Vectorize
        text_tfidf = self.tfidf_vectorizer.transform([processed_text])

        # Predict
        prediction = self.best_model.predict(text_tfidf)[0]
        probability = self.best_model.predict_proba(text_tfidf)[0]

        # Decode label if necessary
        if self.label_encoder:
            predicted_label = self.label_encoder.inverse_transform([prediction])[
                0]
        else:
            predicted_label = prediction

        return predicted_label, max(probability)

    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        print("Starting Amazon Sentiment Analysis Pipeline...")
        print("="*60)

        # Load data
        df = self.load_data()

        # Explore data
        df = self.explore_data(df)

        # Preprocess data
        df = self.preprocess_data(df)

        # Prepare features
        X, y = self.prepare_features(df)

        # Train models
        X_test, y_test, predictions = self.train_models(X, y)

        # Save model
        self.save_model()

        # Test model
        self.test_model()

        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best Model: {self.model_name}")
        print("Ready for deployment!")


if __name__ == "__main__":
    # Run the complete pipeline
    analyzer = AmazonSentimentAnalyzer()
    analyzer.run_complete_pipeline()
