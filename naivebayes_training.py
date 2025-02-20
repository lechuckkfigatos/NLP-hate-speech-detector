import pandas as pd
import string
import math
from collections import defaultdict
import pickle
from vocabulary import build_vocab

def train_naive_bayes(texts, labels, target_classes, alpha=1):
    """
    Huấn luyện mô hình Multinomial Naive Bayes.
    """
    ndoc = 0
    nc = defaultdict(int)
    logprior = dict()
    loglikelihood = dict()
    count = defaultdict(int)

    # Xây dựng từ điển
    vocab = build_vocab(texts)

    # Đếm số lần từ xuất hiện trong từng lớp
    for s, c in zip(texts, labels):
        ndoc += 1
        nc[c] += 1
        for w in s.split():
            if w in vocab:
                count[(w, c)] += 1

    # Tính log-prior và log-likelihood
    vocab_size = len(vocab)
    for c in target_classes:
        logprior[c] = math.log(nc[c] / ndoc)
        sum_wc = sum(count[(w, c)] for w in vocab)

        for w in vocab:
            loglikelihood[(w, c)] = math.log((count[(w, c)] + alpha) / (sum_wc + alpha * vocab_size))

    return logprior, loglikelihood, vocab

if __name__ == "__main__":
    # 1. Đọc dữ liệu từ file CSV
    train_path = "train_data.csv"  # Thay đổi đường dẫn cho đúng file của bạn
    df = pd.read_csv(train_path)

    # 2. Trích xuất cột 'text' và 'label'
    texts = df['Content'].tolist()  # Danh sách văn bản

    # Correctly read labels and convert to integers
    labels = df['Label'].astype(int).tolist()

    # 3. Danh sách lớp mục tiêu
    target_classes = list(set(labels))  # Các lớp trong dữ liệu

    # 4. Huấn luyện mô hình
    logprior, loglikelihood, vocab = train_naive_bayes(texts, labels, target_classes, alpha=1)

    # Save the trained model
    model = {"logprior": logprior, "loglikelihood": loglikelihood, "vocab": vocab}
    with open("naive_bayes_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved successfully to naive_bayes_model.pkl")