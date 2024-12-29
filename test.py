import math
from collections import Counter
from naivebayes_training import train_naive_bayes
from vocabulary import  build_vocab
# Small test dataset for validation
texts = [
    "hate hate speech",           # Class 0 (hate speech)
    "offensive speech offensive", # Class 1 (offensive)
    "this is neutral",            # Class 2 (neutral)
]
labels = [0, 1, 2]  # 0: hate speech, 1: offensive, 2: neutral
target_classes = [0, 1, 2]  # List of classes

vocab = build_vocab(texts)
# Train the Naive Bayes model
logprior, loglikelihood, vocab = train_naive_bayes(texts, labels, target_classes, alpha=1)

# 1. ASSERTIONS FOR LOG-PRIOR
total_docs = len(texts)
assert math.isclose(logprior[0], math.log(1 / total_docs)), "Log-Prior for Class 0 is incorrect!"
assert math.isclose(logprior[1], math.log(1 / total_docs)), "Log-Prior for Class 1 is incorrect!"
assert math.isclose(logprior[2], math.log(1 / total_docs)), "Log-Prior for Class 2 is incorrect!"

# 2. ASSERTIONS FOR LOG-LIKELIHOOD
# Manually count occurrences of words for each class
word_counts_0 = Counter("hate hate speech".split())  # Class 0
word_counts_1 = Counter("offensive speech offensive".split())  # Class 1
word_counts_2 = Counter("this is neutral".split())  # Class 2

total_words_0 = sum(word_counts_0.values())
total_words_1 = sum(word_counts_1.values())
total_words_2 = sum(word_counts_2.values())

vocab_size = len(vocab)
alpha = 1  # Laplace smoothing

# Expected log-likelihood values for specific words
expected_hate_0 = math.log((word_counts_0['hate'] + alpha) / (total_words_0 + alpha * vocab_size))
expected_speech_1 = math.log((word_counts_1['speech'] + alpha) / (total_words_1 + alpha * vocab_size))
expected_neutral_2 = math.log((word_counts_2['neutral'] + alpha) / (total_words_2 + alpha * vocab_size))

# Assertions for log-likelihood
assert math.isclose(loglikelihood[('hate', 0)], expected_hate_0), "Log-Likelihood for 'hate' in Class 0 is incorrect!"
assert math.isclose(loglikelihood[('speech', 1)], expected_speech_1), "Log-Likelihood for 'speech' in Class 1 is incorrect!"
assert math.isclose(loglikelihood[('neutral', 2)], expected_neutral_2), "Log-Likelihood for 'neutral' in Class 2 is incorrect!"

print("All assertions passed! Your implementation of Naive Bayes is correct.")
