import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# 1. Tải dữ liệu từ file CSV
file_path = "cleaned_hate_speech_data.csv"  # Đường dẫn file đã tiền xử lý
df = pd.read_csv(file_path)

# 3. Tách văn bản và nhãn
texts = df['text'].tolist()  # Danh sách văn bản
labels = df['label'].tolist()  # Danh sách nhãn

# 4. Chia dữ liệu thành train/test với tỷ lệ 80/20
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 5. Thống kê phân phối nhãn trong tập train và test
print(Counter(train_labels))

print(Counter(test_labels))

