import re
import html
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download resources (only needed once)
# nltk.download('wordnet')
# nltk.download('stopwords')

# Stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def load_and_preprocess_data(file_path, output_path):
    """
    Loads, preprocesses, and saves the hate speech dataset.

    Args:
        file_path: Path to the dataset file (CSV).
        output_path: Path to save the cleaned dataset (CSV).
    """
    try:
        # Read CSV, treating the first row as the header
        df = pd.read_csv(file_path, header=0, usecols=['Content', 'Label'])

        # Preprocess data
        df['Content'] = df['Content'].apply(preprocess_text)

        # Remove rows with empty text after preprocessing
        df = df[df['Content'].str.strip() != '']

        # Save cleaned data with original column names and order
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        print(f"Number of rows processed: {len(df)}")
        print("Sample data:\n", df.head())

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def preprocess_text(text):
    """
    Preprocesses text (same as before).
    """
    if not isinstance(text, str):
        return ""

    # Remove mentions (@user)
    text = re.sub(r"(@[A-Za-z0-9_]+)", "", text)

    # Remove hashtags (#) but keep the word after #
    text = re.sub(r"#", "", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Convert HTML entities
    text = html.unescape(text)

    # Remove special characters and numbers, keep letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Lowercase
    text = text.lower()

    # Remove stop words
    text = " ".join(word for word in text.split() if word not in stop_words)

    # Lemmatization
    text = " ".join(lemmatizer.lemmatize(word) for word in text.split())

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove RT (retweet) at the beginning
    text = re.sub(r"^rt", "", text).strip()

    return text

def clean_input_sentence(text):
    """
    Cleans a single input sentence using the same preprocessing steps as before.

    Args:
        text: The input sentence.

    Returns:
        A list of cleaned words from the sentence.
    """
    if not isinstance(text, str):
        return []

    # Remove mentions (@user)
    text = re.sub(r"(@[A-Za-z0-9_]+)", "", text)

    # Remove hashtags (#) but keep the word after #
    text = re.sub(r"#", "", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Convert HTML entities
    text = html.unescape(text)

    # Remove special characters and numbers, keep letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Lowercase
    text = text.lower()

    # Remove stop words
    words = [word for word in text.split() if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", " ".join(words)).strip()

    # Remove RT (retweet) at the beginning
    text = re.sub(r"^rt", "", text).strip()

    return text.split()

if __name__ == "__main__":
    input_file1 = "labeled_data.csv"
    output_file1 = "cleaned_labeled_data_1.csv"
    load_and_preprocess_data(input_file1, output_file1)

    input_file2 = "labeled_data_2.csv"
    output_file2 = "cleaned_labeled_data_2.csv"
    load_and_preprocess_data(input_file2, output_file2)