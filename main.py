import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import pickle
from data_loader import clean_input_sentence
import math

# Load trained model
with open("naive_bayes_model.pkl", "rb") as f:
    model = pickle.load(f)

logprior = model["logprior"]
loglikelihood = model["loglikelihood"]
vocab = model["vocab"]

# Load data
test_file = "test_data.csv"
df_test = pd.read_csv(test_file)

df_test['Label'] = df_test['Label'].astype(int)  # Correct: Convert to integer!

test_texts = df_test['Content'].tolist()
test_labels = df_test['Label'].tolist()

# This main file will predict classes from "cleaned_labeled_data.csv" file, so your
# labels can be [positive, neutral, negative]
target_classes = list(set(test_labels)) # target_classes should be [-1, 0, 1]

print("Target Classes:", target_classes)
print("logprior keys:", logprior.keys())  # CRITICAL: Check this output

def predict(texts, logprior, loglikelihood, target_classes, vocab, alpha=1e-10):
    """Predicts class labels."""
    predictions = []
    for text in texts:
        cleaned_words = clean_input_sentence(text)
        logprob = {}
        for c in target_classes:
            try:
                logprob[c] = logprior[c] # If keys in logprior are integers
            except KeyError:
                print(f"Warning: Class label {c} not found in logprior. Assigning a default value.")
                logprob[c] = -1000

        for word in cleaned_words:
            if word in vocab:
                for c in target_classes:
                    logprob[c] += loglikelihood.get((word, c), math.log(alpha))

        predicted_label = max(logprob, key=logprob.get)
        predictions.append(predicted_label)

    return predictions


predictions = predict(test_texts, logprior, loglikelihood, target_classes, vocab)

# Print predictions, true labels and input text
for text, true_label, pred_label in zip(test_texts, test_labels, predictions):
    print(f"{text} || True Label: {true_label} || Predicted Label: {pred_label}")

# Evaluate
accuracy = accuracy_score(test_labels, predictions)
report = classification_report(test_labels, predictions)

print(f"\nAccuracy: {accuracy}")
print("Classification Report:")
print(report)

# Corrected label_mapping
label_mapping = {
    -1: "Negative",
    0: "Neutral",
    1: "Positive"
}


while True:
    user_input = input("Enter a sentence (or type 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break

    cleaned_input = clean_input_sentence(user_input)
    predictions = predict([" ".join(cleaned_input)], logprior, loglikelihood, target_classes, vocab)
    predicted_label = predictions[0]

    # Get description from mapping
    label_description = label_mapping.get(predicted_label, "Unknown")  # Use integer key

    print(f"Predicted label: {predicted_label} ({label_description})")