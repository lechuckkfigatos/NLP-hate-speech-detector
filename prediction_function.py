def test_naive_bayes(testdoc, logprior, loglikelihood, target_classes, vocab):
    """
    Predict the class for a given document using Naive Bayes.

    Args:
        testdoc (str): The input document.
        logprior (dict): Log-prior probabilities for each class.
        loglikelihood (dict): Log-likelihood of words for each class.
        target_classes (list): List of all possible classes.
        vocab (dict): Vocabulary built during training.

    Returns:
        predicted_class: The class with the highest probability.
    """
    sum_ = {}  # Store log-probabilities for each class

    for c in target_classes:
        sum_[c] = logprior[c]  # Start with the log-prior of the class

        # Add log-likelihood for each word in the document
        for w in testdoc.split():
            if w in vocab:  # Only consider words in the vocabulary
                sum_[c] += loglikelihood.get((w, c), 0)  # Default to 0 if word not in loglikelihood

    # Sort classes by log-probability and return the best one
    predicted_class = max(sum_, key=sum_.get)
    return predicted_class

# Predict for all test documents
def predict(test_texts, logprior, loglikelihood, target_classes, vocab):
    """
    Predict the class for a list of documents.
    """
    predictions = []
    for doc in test_texts:  # Iterate through each document in the list
        pred = test_naive_bayes(doc, logprior, loglikelihood, target_classes, vocab)
        predictions.append(pred)
    return predictions

