from collections import Counter
import string

def build_vocab(texts, min_freq=2, ngram_range=(1, 1)):
    """
    Builds vocabulary from a list of texts, with options for frequency filtering and n-grams.

    Args:
        texts: List of text data (strings).
        min_freq: Minimum frequency for a word to be included in the vocabulary.
        ngram_range: Tuple (min_n, max_n) specifying the range of n-gram sizes to consider.
                     (1, 1) for unigrams, (1, 2) for unigrams and bigrams, (2, 2) for bigrams only, etc.

    Returns:
        A vocabulary (dictionary) mapping words/n-grams to unique integer indices.
    """
    word_counts = Counter()
    for text in texts:
        # Preprocessing: Lowercase and remove punctuation
        words = [word.lower().strip(string.punctuation) for word in text.split() if word.strip(string.punctuation)]

        # N-gram generation
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                word_counts[ngram] += 1

    # Build vocabulary with frequency filtering
    vocab = {}
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab