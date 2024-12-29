import re
import html
import emoji
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tải tài nguyên cần thiết từ NLTK
import nltk

# chi chay lan dau
nltk.download('wordnet')
nltk.download('stopwords')

# Khởi tạo stopwords và lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def load_hate_speech_data(file_path):
    """
    Loads the hate speech dataset and extracts labels and text.

    Args:
        file_path: Path to the dataset file (CSV).

    Returns:
        A list of tuples, where each tuple contains (label, text).
    """
    data = []
    try:
        df = pd.read_csv(file_path)

        # Tiền xử lý dữ liệu

        for _, row in df.iterrows():
            label = row['class']
            text = preprocess_text(row['tweet'])
            if text:  # Chỉ thêm dữ liệu không rỗng
                data.append((label, text))

        return data
    except FileNotFoundError:
        print(f"File không tồn tại: {file_path}")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")


def preprocess_text(text):
    """
    Tiền xử lý văn bản bằng cách loại bỏ các ký tự đặc biệt, emoji, URL,
    chuyển đổi HTML entities, và lowercase.

    Args:
        text: Chuỗi văn bản đầu vào.

    Returns:
        Chuỗi văn bản đã được tiền xử lý.
    """
    # Xóa mentions (@user)
    text = re.sub(r"(@[A-Za-z0-9_]+)", "", text)

    # Xóa hashtags (#) nhưng giữ từ sau #
    text = re.sub(r"#", "", text)

    # Xóa URL
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Chuyển HTML entities
    text = html.unescape(text)

    # Chuyển emoji thành dạng văn bản
    text = emoji.demojize(text)

    # Xóa ký tự đặc biệt và số, giữ chữ cái và khoảng trắng
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Chuyển thành chữ thường
    text = text.lower()

    # Xóa stopwords
    text = " ".join(word for word in text.split() if word not in stop_words)

    # Lemmatization
    text = " ".join(lemmatizer.lemmatize(word) for word in text.split())

    # Xóa khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Bỏ RT (retweet) ở đầu câu
    text = re.sub(r"^rt", "", text).strip()
    return text


def save_preprocessed_data(data, output_path):
    """
    Lưu dữ liệu đã xử lý vào file.
    """
    df = pd.DataFrame(data, columns=["label", "text"])
    df.to_csv(output_path, index=False)
    print(f"Dữ liệu đã lưu tại: {output_path}")


if __name__ == "__main__":
    input_file = "labeled_data.csv"
    output_file = "cleaned_hate_speech_data.csv"

    processed_data = load_hate_speech_data(input_file)

    save_preprocessed_data(processed_data, output_file)



