import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from prediction_function import predict
import pickle

# Load the trained model
with open("naive_bayes_model.pkl", "rb") as f:
    model = pickle.load(f)

logprior = model["logprior"]
loglikelihood = model["loglikelihood"]
vocab = model["vocab"]

# Load test data
test_file = "test_data.csv"
df_test = pd.read_csv(test_file)

test_texts = df_test['text'].tolist()
test_labels = df_test['label'].tolist()

# Get target classes
target_classes = list(logprior.keys())

# Predict labels
predictions = predict(test_texts, logprior, loglikelihood, target_classes, vocab)

# Print results
for text, true_label, pred_label in zip(test_texts, test_labels, predictions):
    print(f"{text} || True Label: {true_label} || Predicted Label: {pred_label}")

# Evaluate model

accuracy = accuracy_score(test_labels, predictions)
report = classification_report(test_labels, predictions)

print(f"\nAccuracy: {accuracy}")
print("Classification Report:")
print(report)

