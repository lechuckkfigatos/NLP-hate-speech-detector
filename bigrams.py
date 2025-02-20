from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load your data
df = pd.read_csv("cleaned_hate_speech_data0.csv")
texts = df['text'].tolist()
labels = df['label'].tolist()

# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Create a pipeline with TF-IDF and Naive Bayes
# ngram_range=(1, 2) means use unigrams (single words) and bigrams
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    MultinomialNB()
)

# Train the model
model.fit(train_texts, train_labels)

# Make predictions
predictions = model.predict(test_texts)

# Evaluate
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy with bigrams and TF-IDF: {accuracy}")