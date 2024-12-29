import pandas as pd
import string



# 1. Tải dữ liệu từ file CSV
file_path = "cleaned_hate_speech_data.csv"  # Đường dẫn file
df = pd.read_csv(file_path)

# 2. Lấy danh sách văn bản từ cột 'text'
texts = df['text'].tolist()

def build_vocab(texts):
    """Build vocabulary from dataset.

    Args:
        texts (list): list of tokenized sentences.

    Returns:
        vocab (dict): map from word to index.
    """
    vocab = {}
    for s in texts:
        for word in s.split():
            # Loại bỏ dấu câu (punctuation)
            if word in string.punctuation:
                continue
            # Thêm từ vào vocab nếu chưa có
            if word not in vocab:
                idx = len(vocab)
                vocab[word] = idx
                print(vocab)
    return vocab


